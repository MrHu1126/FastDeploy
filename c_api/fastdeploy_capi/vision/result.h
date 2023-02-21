// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "fastdeploy_capi/fd_common.h"
#include "fastdeploy_capi/fd_type.h"

typedef struct FD_C_ClassifyResultWrapper FD_C_ClassifyResultWrapper;
typedef struct FD_C_DetectionResultWrapper FD_C_DetectionResultWrapper;
typedef struct FD_C_OCRResultWrapper FD_C_OCRResultWrapper;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FD_C_ClassifyResult {
  FD_C_OneDimArrayInt32 label_ids;
  FD_C_OneDimArrayFloat scores;
  FD_C_ResultType type;
} FD_C_ClassifyResult;

typedef struct FD_C_OneDimClassifyResult {
  size_t size;
  FD_C_ClassifyResult* data;
} FD_C_OneDimClassifyResult;

typedef struct FD_C_Mask {
  FD_C_OneDimArrayUint8 data;
  FD_C_OneDimArrayInt64 shape;
  FD_C_ResultType type;
} FD_C_Mask;

typedef struct FD_C_OneDimMask {
  size_t size;
  FD_C_Mask* data;
} FD_C_OneDimMask;  // std::vector<FD_C_Mask>

typedef struct FD_C_DetectionResult {
  FD_C_TwoDimArrayFloat boxes;
  FD_C_OneDimArrayFloat scores;
  FD_C_OneDimArrayInt32 label_ids;
  FD_C_OneDimMask masks;
  FD_C_Bool contain_masks;
  FD_C_ResultType type;
} FD_C_DetectionResult;

typedef struct FD_C_OneDimDetectionResult {
  size_t size;
  FD_C_DetectionResult* data;
} FD_C_OneDimDetectionResult;


typedef struct FD_C_OCRResult {
  FD_C_TwoDimArrayInt32 boxes;
  FD_C_OneDimArrayCstr text;
  FD_C_OneDimArrayFloat rec_scores;
  FD_C_OneDimArrayFloat cls_scores;
  FD_C_OneDimArrayInt32 cls_labels;
  FD_C_ResultType type;
} FD_C_OCRResult;

typedef struct FD_C_OneDimOCRResult {
  size_t size;
  FD_C_OCRResult* data;
} FD_C_OneDimOCRResult;

// Classification Results

/** \brief Create a new FD_C_ClassifyResultWrapper object
 *
 * \return Return a pointer to FD_C_ClassifyResultWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_ClassifyResultWrapper*
FD_C_CreateClassifyResultWrapper();

/** \brief Destroy a FD_C_ClassifyResultWrapper object
 *
 * \param[in] fd_c_classify_result_wrapper pointer to FD_C_ClassifyResultWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern void FD_C_DestroyClassifyResultWrapper(
    __fd_take FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper);

/** \brief Destroy a FD_C_ClassifyResult object
 *
 * \param[in] fd_c_classify_result pointer to FD_C_ClassifyResult object
 */

FASTDEPLOY_CAPI_EXPORT extern void
FD_C_DestroyClassifyResult(__fd_take FD_C_ClassifyResult* fd_c_classify_result);

/** \brief Get a FD_C_ClassifyResult object from FD_C_ClassifyResultWrapper object
 *
 * \param[in] fd_c_classify_result_wrapper pointer to FD_C_ClassifyResultWrapper object
 * \return Return a pointer to FD_C_ClassifyResult object
 */
FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_ClassifyResult*
FD_C_ClassifyResultWrapperGetData(
    __fd_keep FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper);

/** \brief Create a new FD_C_ClassifyResultWrapper object from FD_C_ClassifyResult object
 *
 * \param[in] fd_c_classify_result pointer to FD_C_ClassifyResult object
 * \return Return a pointer to FD_C_ClassifyResultWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_ClassifyResultWrapper*
FD_C_CreateClassifyResultWrapperFromData(
    __fd_keep FD_C_ClassifyResult* fd_c_classify_result);

/** \brief Print ClassifyResult formated information
 *
 * \param[in] fd_c_classify_result_wrapper pointer to FD_C_ClassifyResultWrapper object
 * \return Return a string pointer
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give char*
FD_C_ClassifyResultWrapperStr(
    __fd_keep FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper);

// Detection Results

/** \brief Create a new FD_C_DetectionResultWrapper object
 *
 * \return Return a pointer to FD_C_DetectionResultWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_DetectionResultWrapper*
FD_C_CreateDetectionResultWrapper();

/** \brief Destroy a FD_C_DetectionResultWrapper object
 *
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern void FD_C_DestroyDetectionResultWrapper(
    __fd_take FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper);

/** \brief Destroy a FD_C_DetectionResult object
 *
 * \param[in] fd_c_detection_result pointer to FD_C_DetectionResult object
 */

FASTDEPLOY_CAPI_EXPORT extern void FD_C_DestroyDetectionResult(
    __fd_take FD_C_DetectionResult* fd_c_detection_result);

/** \brief Get a FD_C_DetectionResult object from FD_C_DetectionResultWrapper object
 *
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object
 * \return Return a pointer to FD_C_DetectionResult object
 */
FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_DetectionResult*
FD_C_DetectionResultWrapperGetData(
    __fd_keep FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper);

/** \brief Create a new FD_C_DetectionResultWrapper object from FD_C_DetectionResult object
 *
 * \param[in] fd_c_detection_result pointer to FD_C_DetectionResult object
 * \return Return a pointer to FD_C_DetectionResultWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_DetectionResultWrapper*
FD_C_CreateDetectionResultWrapperFromData(
    __fd_keep FD_C_DetectionResult* fd_c_detection_result);

/** \brief Print DetectionResult formated information
 *
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object
 * \return Return a string pointer
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give char*
FD_C_DetectionResultWrapperStr(
    __fd_keep FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper);


// OCR Results

/** \brief Create a new FD_C_OCRResultWrapper object
 *
 * \return Return a pointer to FD_C_OCRResultWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_OCRResultWrapper*
FD_C_CreateOCRResultWrapper();

/** \brief Destroy a FD_C_OCRResultWrapper object
 *
 * \param[in] fd_c_ocr_result_wrapper pointer to FD_C_OCRResultWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern void FD_C_DestroyOCRResultWrapper(
    __fd_take FD_C_OCRResultWrapper* fd_c_ocr_result_wrapper);

/** \brief Destroy a FD_C_OCRResult object
 *
 * \param[in] fd_c_ocr_result pointer to FD_C_OCRResult object
 */

FASTDEPLOY_CAPI_EXPORT extern void FD_C_DestroyOCRResult(
    __fd_take FD_C_OCRResult* fd_c_ocr_result);

/** \brief Get a FD_C_OCRResult object from FD_C_OCRResultWrapper object
 *
 * \param[in] fd_c_ocr_result_wrapper pointer to FD_C_OCRResultWrapper object
 * \return Return a pointer to FD_C_OCRResult object
 */
FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_OCRResult*
FD_C_OCRResultWrapperGetData(
    __fd_keep FD_C_OCRResultWrapper* fd_c_ocr_result_wrapper);

/** \brief Create a new FD_C_OCRResultWrapper object from FD_C_OCRResult object
 *
 * \param[in] fd_c_ocr_result pointer to FD_C_OCRResult object
 * \return Return a pointer to FD_C_OCRResultWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_OCRResultWrapper*
FD_C_CreateOCRResultWrapperFromData(
    __fd_keep FD_C_OCRResult* fd_c_ocr_result);

/** \brief Print OCRResult formated information
 *
 * \param[in] fd_c_ocr_result_wrapper pointer to FD_C_OCRResultWrapper object
 * \return Return a string pointer
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give char*
FD_C_OCRResultWrapperStr(
    __fd_keep FD_C_OCRResultWrapper* fd_c_ocr_result_wrapper);


#ifdef __cplusplus
}  // extern "C"
#endif
