# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fastdeploy as fd
import cv2
import os
import numpy as np
import time


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True, help="Model dir of PPOCR.")
    parser.add_argument(
        "--det_model", required=True, help="Path of Detection model of PPOCR.")
    parser.add_argument(
        "--cls_model",
        required=True,
        help="Path of Classification model of PPOCR.")
    parser.add_argument(
        "--rec_model",
        required=True,
        help="Path of Recognization model of PPOCR.")
    parser.add_argument(
        "--rec_label_file",
        required=True,
        help="Path of Recognization model of PPOCR.")
    parser.add_argument(
        "--image", type=str, required=False, help="Path of test image file.")
    parser.add_argument(
        "--cpu_num_thread",
        type=int,
        default=8,
        help="default number of cpu thread.")
    parser.add_argument(
        "--device_id", type=int, default=0, help="device(gpu) id")
    parser.add_argument(
        "--iter_num",
        required=True,
        type=int,
        default=300,
        help="number of iterations for computing performace.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--backend",
        type=str,
        default="default",
        help="inference backend, default, ort, ov, trt, paddle, paddle_trt.")
    parser.add_argument(
        "--enable_trt_fp16",
        type=ast.literal_eval,
        default=False,
        help="whether enable fp16 in trt backend")
    parser.add_argument(
        "--enable_collect_memory_info",
        type=ast.literal_eval,
        default=False,
        help="whether enable collect memory info")
    args = parser.parse_args()
    return args


def build_option(args):
    option = fd.RuntimeOption()
    device = args.device
    backend = args.backend
    enable_trt_fp16 = args.enable_trt_fp16
    option.set_cpu_thread_num(args.cpu_num_thread)
    if device == "gpu":
        option.use_gpu()
        if backend == "ort":
            option.use_ort_backend()
        elif backend == "paddle":
            option.use_paddle_backend()
        elif backend in ["trt", "paddle_trt"]:
            option.use_trt_backend()
            if backend == "paddle_trt":
                option.enable_paddle_to_trt()
            if enable_trt_fp16:
                option.enable_trt_fp16()
        elif backend == "default":
            return option
        else:
            raise Exception(
                "While inference with GPU, only support default/ort/paddle/trt/paddle_trt now, {} is not supported.".
                format(backend))
    elif device == "cpu":
        if backend == "ort":
            option.use_ort_backend()
        elif backend == "ov":
            option.use_openvino_backend()
        elif backend == "paddle":
            option.use_paddle_backend()
        elif backend == "default":
            return option
        else:
            raise Exception(
                "While inference with CPU, only support default/ort/ov/paddle now, {} is not supported.".
                format(backend))
    else:
        raise Exception(
            "Only support device CPU/GPU now, {} is not supported.".format(
                device))

    return option


def get_current_memory_mb(gpu_id=None):
    import pynvml
    import psutil
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    cpu_mem = info.uss / 1024. / 1024.
    gpu_mem = 0
    if gpu_id is not None:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = meminfo.used / 1024. / 1024.
    return cpu_mem, gpu_mem


def get_current_gputil(gpu_id):
    import GPUtil
    GPUs = GPUtil.getGPUs()
    gpu_load = GPUs[gpu_id].load
    return gpu_load


if __name__ == '__main__':

    args = parse_arguments()
    option = build_option(args)
    # Detection Model
    det_model_file = os.path.join(args.model_dir, args.det_model,
                                  "inference.pdmodel")
    det_params_file = os.path.join(args.model_dir, args.det_model,
                                   "inference.pdiparams")
    # Classification Model
    cls_model_file = os.path.join(args.model_dir, args.cls_model,
                                  "inference.pdmodel")
    cls_params_file = os.path.join(args.model_dir, args.cls_model,
                                   "inference.pdiparams")
    # Recognition Model
    rec_model_file = os.path.join(args.model_dir, args.rec_model,
                                  "inference.pdmodel")
    rec_params_file = os.path.join(args.model_dir, args.rec_model,
                                   "inference.pdiparams")
    rec_label_file = os.path.join(args.model_dir, args.rec_label_file)

    gpu_id = args.device_id
    enable_collect_memory_info = args.enable_collect_memory_info
    end2end_statis = list()
    cpu_mem = list()
    gpu_mem = list()
    gpu_util = list()
    if args.device == "cpu":
        file_path = args.model_dir + "_model_" + args.backend + "_" + \
            args.device + "_" + str(args.cpu_num_thread) + ".txt"
    else:
        if args.enable_trt_fp16:
            file_path = args.model_dir + "_model_" + args.backend + "_fp16_" + args.device + ".txt"
        else:
            file_path = args.model_dir + "_model_" + args.backend + "_" + args.device + ".txt"
    f = open(file_path, "w")
    f.writelines("===={}====: \n".format(os.path.split(file_path)[-1][:-4]))

    try:
        rec_option = option
        if "OCRv2" in args.model_dir:
            det_option = option
            if args.backend in ["trt", "paddle_trt"]:
                det_option.set_trt_input_shape(
                    "x", [1, 3, 64, 64], [1, 3, 640, 640], [1, 3, 960, 960])
            det_model = fd.vision.ocr.DBDetector(
                det_model_file, det_params_file, runtime_option=det_option)
            cls_option = option
            if args.backend in ["trt", "paddle_trt"]:
                cls_option.set_trt_input_shape(
                    "x", [1, 3, 48, 10], [10, 3, 48, 320], [64, 3, 48, 1024])
            cls_model = fd.vision.ocr.Classifier(
                cls_model_file, cls_params_file, runtime_option=cls_option)
            rec_option = option
            if args.backend in ["trt", "paddle_trt"]:
                rec_option.set_trt_input_shape(
                    "x", [1, 3, 32, 10], [10, 3, 32, 320], [32, 3, 32, 2304])
            rec_model = fd.vision.ocr.Recognizer(
                rec_model_file,
                rec_params_file,
                rec_label_file,
                runtime_option=rec_option)
            model = fd.vision.ocr.PPOCRv2(
                det_model=det_model, cls_model=cls_model, rec_model=rec_model)
        elif "OCRv3" in args.model_dir:
            if args.backend in ["trt", "paddle_trt"]:
                det_option.set_trt_input_shape(
                    "x", [1, 3, 64, 64], [1, 3, 640, 640], [1, 3, 960, 960])
            det_model = fd.vision.ocr.DBDetector(
                det_model_file, det_params_file, runtime_option=det_option)
            if args.backend in ["trt", "paddle_trt"]:
                cls_option.set_trt_input_shape(
                    "x", [1, 3, 48, 10], [10, 3, 48, 320], [64, 3, 48, 1024])
            cls_model = fd.vision.ocr.Classifier(
                cls_model_file, cls_params_file, runtime_option=cls_option)
            if args.backend in ["trt", "paddle_trt"]:
                rec_option.set_trt_input_shape(
                    "x", [1, 3, 48, 10], [10, 3, 48, 320], [64, 3, 48, 2304])
            rec_model = fd.vision.ocr.Recognizer(
                rec_model_file,
                rec_params_file,
                rec_label_file,
                runtime_option=rec_option)
            model = fd.vision.ocr.PPOCRv3(
                det_model=det_model, cls_model=cls_model, rec_model=rec_model)
        else:
            raise Exception("model {} not support now in ppocr series".format(
                args.model_dir))
        det_model.enable_record_time_of_runtime()
        cls_model.enable_record_time_of_runtime()
        rec_model.enable_record_time_of_runtime()
        im_ori = cv2.imread(args.image)
        for i in range(args.iter_num):
            im = im_ori
            start = time.time()
            result = model.predict(im)
            end2end_statis.append(time.time() - start)
            if enable_collect_memory_info:
                gpu_util.append(get_current_gputil(gpu_id))
                cm, gm = get_current_memory_mb(gpu_id)
                cpu_mem.append(cm)
                gpu_mem.append(gm)

        runtime_statis_det = det_model.print_statis_info_of_runtime()
        runtime_statis_cls = cls_model.print_statis_info_of_runtime()
        runtime_statis_rec = rec_model.print_statis_info_of_runtime()

        warmup_iter = args.iter_num // 5
        end2end_statis_repeat = end2end_statis[warmup_iter:]
        if enable_collect_memory_info:
            cpu_mem_repeat = cpu_mem[warmup_iter:]
            gpu_mem_repeat = gpu_mem[warmup_iter:]
            gpu_util_repeat = gpu_util[warmup_iter:]

        dump_result = dict()
        dump_result["runtime"] = (
            runtime_statis_det["avg_time"] + runtime_statis_cls["avg_time"] +
            runtime_statis_rec["avg_time"]) * 1000
        dump_result["end2end"] = np.mean(end2end_statis_repeat) * 1000
        if enable_collect_memory_info:
            dump_result["cpu_rss_mb"] = np.mean(cpu_mem_repeat)
            dump_result["gpu_rss_mb"] = np.mean(gpu_mem_repeat)
            dump_result["gpu_util"] = np.mean(gpu_util_repeat)

        f.writelines("Runtime(ms): {} \n".format(str(dump_result["runtime"])))
        f.writelines("End2End(ms): {} \n".format(str(dump_result["end2end"])))
        if enable_collect_memory_info:
            f.writelines("cpu_rss_mb: {} \n".format(
                str(dump_result["cpu_rss_mb"])))
            f.writelines("gpu_rss_mb: {} \n".format(
                str(dump_result["gpu_rss_mb"])))
            f.writelines("gpu_util: {} \n".format(
                str(dump_result["gpu_util"])))
    except:
        f.writelines("!!!!!Infer Failed\n")

    f.close()
