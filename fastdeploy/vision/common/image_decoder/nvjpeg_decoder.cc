/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fastdeploy/vision/common/image_decoder/nvjpeg_decoder.h"

namespace fastdeploy {
namespace vision {
namespace nvjpeg {

#define CHECK_CUDA(call)                                                   \
  {                                                                        \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
      std::cout << "CUDA Runtime failure: '#" << _e << "' at " << __FILE__ \
                << ":" << __LINE__ << std::endl;                           \
      exit(1);                                                             \
    }                                                                      \
  }

#define CHECK_NVJPEG(call)                                                  \
  {                                                                         \
    nvjpegStatus_t _e = (call);                                             \
    if (_e != NVJPEG_STATUS_SUCCESS) {                                      \
      std::cout << "NVJPEG failure: '#" << _e << "' at " << __FILE__ << ":" \
                << __LINE__ << std::endl;                                   \
      exit(1);                                                              \
    }                                                                       \
  }

static int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }

static int dev_free(void *p) { return (int)cudaFree(p); }

static int host_malloc(void **p, size_t s, unsigned int f) {
  return (int)cudaHostAlloc(p, s, f);
}

static int host_free(void *p) { return (int)cudaFreeHost(p); }

static int read_images(const FileNames &image_names, FileData &raw_data,
                       std::vector<size_t> &raw_len) {
  // int counter = 0;

  // while (counter < batch_size)
  // {
  //     if (cur_iter == image_names.end())
  //     {
  //         std::cerr << "Image list is too short to fill the batch, adding
  //         files "
  //                      "from the beginning of the image list"
  //                   << std::endl;
  //         cur_iter = image_names.begin();
  //     }

  for (size_t i = 0; i < image_names.size(); ++i) {
    if (image_names.size() == 0) {
      std::cerr << "No valid images left in the input list, exit" << std::endl;
      return EXIT_FAILURE;
    }

    // Read an image from disk.
    std::ifstream input(image_names[i].c_str(),
                        std::ios::in | std::ios::binary | std::ios::ate);
    if (!(input.is_open())) {
      std::cerr << "Cannot open image: " << image_names[i] << std::endl;
      FDASSERT(false, "Read file error.");
      continue;
    }

    // Get the size
    long unsigned int file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    // resize if buffer is too small
    if (raw_data[i].size() < file_size) {
      raw_data[i].resize(file_size);
    }
    if (!input.read(raw_data[i].data(), file_size)) {
      std::cerr << "Cannot read from file: " << image_names[i] << std::endl;
      // image_names.erase(cur_iter);
      FDASSERT(false, "Read file error.");
      continue;
    }
    raw_len[i] = file_size;
  }
  return EXIT_SUCCESS;
}

// prepare buffers for RGBi output format
static int prepare_buffers(FileData &file_data, std::vector<size_t> &file_len,
                           std::vector<int> &img_width,
                           std::vector<int> &img_height,
                           std::vector<nvjpegImage_t> &ibuf,
                           std::vector<nvjpegImage_t> &isz,
                           std::vector<FDTensor *> &output_buffers,
                           const FileNames &current_names,
                           decode_params_t &params) {
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int channels;
  nvjpegChromaSubsampling_t subsampling;

  for (long unsigned int i = 0; i < file_data.size(); i++) {
    CHECK_NVJPEG(nvjpegGetImageInfo(
        params.nvjpeg_handle, (unsigned char *)file_data[i].data(), file_len[i],
        &channels, &subsampling, widths, heights));

    img_width[i] = widths[0];
    img_height[i] = heights[0];
    std::cout << "Processing: " << current_names[i] << std::endl;
#ifdef NVJPEG_DEBUG
    std::cout << "Image is " << channels << " channels." << std::endl;
    for (int c = 0; c < channels; c++) {
      std::cout << "Channel #" << c << " size: " << widths[c] << " x "
                << heights[c] << std::endl;
    }
    switch (subsampling) {
      case NVJPEG_CSS_444:
        std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
        break;
      case NVJPEG_CSS_440:
        std::cout << "YUV 4:4:0 chroma subsampling" << std::endl;
        break;
      case NVJPEG_CSS_422:
        std::cout << "YUV 4:2:2 chroma subsampling" << std::endl;
        break;
      case NVJPEG_CSS_420:
        std::cout << "YUV 4:2:0 chroma subsampling" << std::endl;
        break;
      case NVJPEG_CSS_411:
        std::cout << "YUV 4:1:1 chroma subsampling" << std::endl;
        break;
      case NVJPEG_CSS_410:
        std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
        break;
      case NVJPEG_CSS_GRAY:
        std::cout << "Grayscale JPEG " << std::endl;
        break;
      case NVJPEG_CSS_410V:
        std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
        break;
      case NVJPEG_CSS_UNKNOWN:
        std::cout << "Unknown chroma subsampling" << std::endl;
        return EXIT_FAILURE;
    }
#endif

    int mul = 1;
    // in the case of interleaved RGB output, write only to single channel, but
    // 3 samples at once
    if (params.fmt == NVJPEG_OUTPUT_RGBI || params.fmt == NVJPEG_OUTPUT_BGRI) {
      channels = 1;
      mul = 3;
    } else if (params.fmt == NVJPEG_OUTPUT_RGB ||
               params.fmt == NVJPEG_OUTPUT_BGR) {
      // in the case of rgb create 3 buffers with sizes of original image
      channels = 3;
      widths[1] = widths[2] = widths[0];
      heights[1] = heights[2] = heights[0];
    } else {
      FDASSERT(false, "Unsupport NVJPEG output format: %d", params.fmt);
    }

    output_buffers[i]->Resize({heights[0], widths[0], mul * channels},
                              FDDataType::UINT8, "output_cache", Device::GPU);

    uint8_t *cur_buffer =
        reinterpret_cast<uint8_t *>(output_buffers[i]->Data());

    // realloc output buffer if required
    for (int c = 0; c < channels; c++) {
      int aw = mul * widths[c];
      int ah = heights[c];
      size_t sz = aw * ah;
      ibuf[i].pitch[c] = aw;
      if (sz > isz[i].pitch[c]) {
        ibuf[i].channel[c] = cur_buffer;
        cur_buffer = cur_buffer + sz;
        isz[i].pitch[c] = sz;
      }
    }
  }
  return EXIT_SUCCESS;
}

static void create_decoupled_api_handles(decode_params_t &params) {
  CHECK_NVJPEG(nvjpegDecoderCreate(params.nvjpeg_handle, NVJPEG_BACKEND_DEFAULT,
                                   &params.nvjpeg_decoder));
  CHECK_NVJPEG(nvjpegDecoderStateCreate(params.nvjpeg_handle,
                                        params.nvjpeg_decoder,
                                        &params.nvjpeg_decoupled_state));

  CHECK_NVJPEG(nvjpegBufferPinnedCreate(params.nvjpeg_handle, NULL,
                                        &params.pinned_buffers[0]));
  CHECK_NVJPEG(nvjpegBufferPinnedCreate(params.nvjpeg_handle, NULL,
                                        &params.pinned_buffers[1]));
  CHECK_NVJPEG(nvjpegBufferDeviceCreate(params.nvjpeg_handle, NULL,
                                        &params.device_buffer));

  CHECK_NVJPEG(
      nvjpegJpegStreamCreate(params.nvjpeg_handle, &params.jpeg_streams[0]));
  CHECK_NVJPEG(
      nvjpegJpegStreamCreate(params.nvjpeg_handle, &params.jpeg_streams[1]));

  CHECK_NVJPEG(nvjpegDecodeParamsCreate(params.nvjpeg_handle,
                                        &params.nvjpeg_decode_params));
}

static void destroy_decoupled_api_handles(decode_params_t &params) {
  CHECK_NVJPEG(nvjpegDecodeParamsDestroy(params.nvjpeg_decode_params));
  CHECK_NVJPEG(nvjpegJpegStreamDestroy(params.jpeg_streams[0]));
  CHECK_NVJPEG(nvjpegJpegStreamDestroy(params.jpeg_streams[1]));
  CHECK_NVJPEG(nvjpegBufferPinnedDestroy(params.pinned_buffers[0]));
  CHECK_NVJPEG(nvjpegBufferPinnedDestroy(params.pinned_buffers[1]));
  CHECK_NVJPEG(nvjpegBufferDeviceDestroy(params.device_buffer));
  CHECK_NVJPEG(nvjpegJpegStateDestroy(params.nvjpeg_decoupled_state));
  CHECK_NVJPEG(nvjpegDecoderDestroy(params.nvjpeg_decoder));
}

int decode_images(const FileData &img_data, const std::vector<size_t> &img_len,
                  std::vector<nvjpegImage_t> &out, decode_params_t &params,
                  double &time) {
  CHECK_CUDA(cudaStreamSynchronize(params.stream));
  // cudaEvent_t startEvent = NULL, stopEvent = NULL;
  // float       loopTime = 0;

  // CHECK_CUDA(cudaEventCreate(&startEvent));
  // CHECK_CUDA(cudaEventCreate(&stopEvent));

  std::vector<const unsigned char *> batched_bitstreams;
  std::vector<size_t> batched_bitstreams_size;
  std::vector<nvjpegImage_t> batched_output;

  // bit-streams that batched decode cannot handle
  std::vector<const unsigned char *> otherdecode_bitstreams;
  std::vector<size_t> otherdecode_bitstreams_size;
  std::vector<nvjpegImage_t> otherdecode_output;

  if (params.hw_decode_available) {
    for (int i = 0; i < params.batch_size; i++) {
      // extract bitstream meta data to figure out whether a bit-stream can be
      // decoded
      nvjpegJpegStreamParseHeader(params.nvjpeg_handle,
                                  (const unsigned char *)img_data[i].data(),
                                  img_len[i], params.jpeg_streams[0]);
      int isSupported = -1;
      nvjpegDecodeBatchedSupported(params.nvjpeg_handle, params.jpeg_streams[0],
                                   &isSupported);

      if (isSupported == 0) {
        batched_bitstreams.push_back((const unsigned char *)img_data[i].data());
        batched_bitstreams_size.push_back(img_len[i]);
        batched_output.push_back(out[i]);
      } else {
        otherdecode_bitstreams.push_back(
            (const unsigned char *)img_data[i].data());
        otherdecode_bitstreams_size.push_back(img_len[i]);
        otherdecode_output.push_back(out[i]);
      }
    }
  } else {
    for (int i = 0; i < params.batch_size; i++) {
      otherdecode_bitstreams.push_back(
          (const unsigned char *)img_data[i].data());
      otherdecode_bitstreams_size.push_back(img_len[i]);
      otherdecode_output.push_back(out[i]);
    }
  }

  // CHECK_CUDA(cudaEventRecord(startEvent, params.stream));

  if (batched_bitstreams.size() > 0) {
    CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(
        params.nvjpeg_handle, params.nvjpeg_state, batched_bitstreams.size(), 1,
        params.fmt));

    CHECK_NVJPEG(nvjpegDecodeBatched(
        params.nvjpeg_handle, params.nvjpeg_state, batched_bitstreams.data(),
        batched_bitstreams_size.data(), batched_output.data(), params.stream));
  }

  if (otherdecode_bitstreams.size() > 0) {
    CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(params.nvjpeg_decoupled_state,
                                               params.device_buffer));
    int buffer_index = 0;
    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(params.nvjpeg_decode_params,
                                                   params.fmt));
    for (int i = 0; i < params.batch_size; i++) {
      CHECK_NVJPEG(nvjpegJpegStreamParse(params.nvjpeg_handle,
                                         otherdecode_bitstreams[i],
                                         otherdecode_bitstreams_size[i], 0, 0,
                                         params.jpeg_streams[buffer_index]));

      CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(
          params.nvjpeg_decoupled_state, params.pinned_buffers[buffer_index]));

      CHECK_NVJPEG(nvjpegDecodeJpegHost(
          params.nvjpeg_handle, params.nvjpeg_decoder,
          params.nvjpeg_decoupled_state, params.nvjpeg_decode_params,
          params.jpeg_streams[buffer_index]));

      CHECK_CUDA(cudaStreamSynchronize(params.stream));

      CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(
          params.nvjpeg_handle, params.nvjpeg_decoder,
          params.nvjpeg_decoupled_state, params.jpeg_streams[buffer_index],
          params.stream));

      buffer_index = 1 - buffer_index;  // switch pinned buffer in pipeline mode
                                        // to avoid an extra sync

      CHECK_NVJPEG(
          nvjpegDecodeJpegDevice(params.nvjpeg_handle, params.nvjpeg_decoder,
                                 params.nvjpeg_decoupled_state,
                                 &otherdecode_output[i], params.stream));
    }
  }
  // CHECK_CUDA(cudaEventRecord(stopEvent, params.stream));

  // CHECK_CUDA(cudaEventSynchronize(stopEvent));
  // CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
  // time = 0.001 * static_cast<double>(loopTime); // cudaEventElapsedTime
  // returns milliseconds

  return EXIT_SUCCESS;
}

double process_images(const FileNames &image_names, decode_params_t &params,
                      double &total, std::vector<nvjpegImage_t> &iout,
                      std::vector<FDTensor *> &output_buffers,
                      std::vector<int> &widths, std::vector<int> &heights) {
  FDASSERT(image_names.size() == params.batch_size,
           "Number of images and batch size must be equal.");
  // vector for storing raw files and file lengths
  FileData file_data(params.batch_size);
  std::vector<size_t> file_len(params.batch_size);
  FileNames current_names(params.batch_size);
  // we wrap over image files to process total_images of files
  auto file_iter = image_names.begin();

  // stream for decoding
  // CHECK_CUDA(cudaStreamCreateWithFlags(&params.stream,
  // cudaStreamNonBlocking));

  // int total_processed = 0;

  // output buffer sizes, for convenience
  std::vector<nvjpegImage_t> isz(params.batch_size);

  for (long unsigned int i = 0; i < iout.size(); i++) {
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
      iout[i].channel[c] = NULL;
      iout[i].pitch[c] = 0;
      isz[i].pitch[c] = 0;
    }
  }

  // double test_time = 0;
  // int    warmup    = 0;
  // while (total_processed < image_names.size())
  // {
  if (read_images(image_names, file_data, file_len)) return EXIT_FAILURE;

  if (prepare_buffers(file_data, file_len, widths, heights, iout, isz,
                      output_buffers, image_names, params))
    return EXIT_FAILURE;

  double time;
  if (decode_images(file_data, file_len, iout, params, time))
    return EXIT_FAILURE;
  // if (warmup < params.warmup)
  // {
  //     warmup++;
  // }
  // else
  // {
  // total_processed += params.batch_size;
  // test_time += time;
  // }

  // if (params.write_decoded)
  //     write_images(iout, widths, heights, params, current_names);
  // }
  // total = test_time;

  // CHECK_CUDA(cudaStreamDestroy(params.stream));

  return EXIT_SUCCESS;
}

void init_decoder(decode_params_t &params) {
  params.hw_decode_available = true;
  nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
  nvjpegPinnedAllocator_t pinned_allocator = {&host_malloc, &host_free};
  nvjpegStatus_t status =
      nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, &dev_allocator, &pinned_allocator,
                     NVJPEG_FLAGS_DEFAULT, &params.nvjpeg_handle);
  if (status == NVJPEG_STATUS_ARCH_MISMATCH) {
    std::cout << "Hardware Decoder not supported. "
                 "Falling back to default backend"
              << std::endl;
    CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator,
                                &pinned_allocator, NVJPEG_FLAGS_DEFAULT,
                                &params.nvjpeg_handle));
    params.hw_decode_available = false;
  } else {
    CHECK_NVJPEG(status);
  }

  CHECK_NVJPEG(
      nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));

  create_decoupled_api_handles(params);
}

void destroy_decoder(decode_params_t &params) {
  destroy_decoupled_api_handles(params);
  CHECK_NVJPEG(nvjpegJpegStateDestroy(params.nvjpeg_state));
  CHECK_NVJPEG(nvjpegDestroy(params.nvjpeg_handle));
}

int NvDecode(std::string images_dir, int total_images, int batch_size,
             nvjpegOutputFormat_t outputFormat, uint8_t *gpuWorkspace) {
  std::vector<nvjpegImage_t> iout;
  std::vector<int> widths, heights;
  iout.resize(batch_size);
  widths.resize(batch_size);
  heights.resize(batch_size);

  decode_params_t params;
  // params.input_dir           = images_dir;
  params.batch_size = batch_size;
  // params.warmup              = 0;
  params.fmt = outputFormat;
  // params.write_decoded       = false;
  params.hw_decode_available = true;
  // params.total_images        = (total_images / params.batch_size) *
  // params.batch_size; params.output_dir          = "./";

  nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
  nvjpegPinnedAllocator_t pinned_allocator = {&host_malloc, &host_free};
  nvjpegStatus_t status =
      nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, &dev_allocator, &pinned_allocator,
                     NVJPEG_FLAGS_DEFAULT, &params.nvjpeg_handle);
  if (status == NVJPEG_STATUS_ARCH_MISMATCH) {
    std::cout
        << "Hardware Decoder not supported. Falling back to default backend"
        << std::endl;
    CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator,
                                &pinned_allocator, NVJPEG_FLAGS_DEFAULT,
                                &params.nvjpeg_handle));
    params.hw_decode_available = false;
  } else {
    CHECK_NVJPEG(status);
  }

  CHECK_NVJPEG(
      nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));

  create_decoupled_api_handles(params);

  // read source images
  FileNames image_names;
  // readInput(params.input_dir, image_names);
  image_names.push_back("ILSVRC2012_val_00000010.jpeg");

  // double total;
  // if (process_images(image_names, params, total, iout, gpuWorkspace, widths,
  // heights))
  //     return EXIT_FAILURE;

#ifdef PROFILE_SAMPLE
  std::cout << "Total decoding time: " << total << " (s)" << std::endl;
  std::cout << "Avg decoding time per image: " << total / params.total_images
            << " (s)" << std::endl;
  std::cout << "Avg images per sec: " << params.total_images / total
            << std::endl;
  std::cout << "Avg decoding time per batch: "
            << total / ((params.total_images + params.batch_size - 1) /
                        params.batch_size)
            << " (s)" << std::endl;
#endif

  destroy_decoupled_api_handles(params);

  CHECK_NVJPEG(nvjpegJpegStateDestroy(params.nvjpeg_state));
  CHECK_NVJPEG(nvjpegDestroy(params.nvjpeg_handle));

  return EXIT_SUCCESS;
}

}  // namespace nvjpeg
}  // namespace vision
}  // namespace fastdeploy
