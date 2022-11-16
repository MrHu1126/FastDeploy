// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {

void ClassifyResult::Clear() {
  std::vector<int32_t>().swap(label_ids);
  std::vector<float>().swap(scores);
}

std::string ClassifyResult::Str() {
  std::string out;
  out = "ClassifyResult(\nlabel_ids: ";
  for (size_t i = 0; i < label_ids.size(); ++i) {
    out = out + std::to_string(label_ids[i]) + ", ";
  }
  out += "\nscores: ";
  for (size_t i = 0; i < label_ids.size(); ++i) {
    out = out + std::to_string(scores[i]) + ", ";
  }
  out += "\n)";
  return out;
}

ClassifyResult& ClassifyResult::operator=(ClassifyResult&& other) {
  if (&other != this) {
    label_ids = std::move(other.label_ids);
    scores = std::move(other.scores);
  }
  return *this;
}

void Mask::Reserve(int size) { data.reserve(size); }

void Mask::Resize(int size) { data.resize(size); }

void Mask::Clear() {
  std::vector<int32_t>().swap(data);
  std::vector<int64_t>().swap(shape);
}

std::string Mask::Str() {
  std::string out = "Mask(";
  size_t ndim = shape.size();
  for (size_t i = 0; i < ndim; ++i) {
    if (i < ndim - 1) {
      out += std::to_string(shape[i]) + ",";
    } else {
      out += std::to_string(shape[i]);
    }
  }
  out += ")\n";
  return out;
}

DetectionResult::DetectionResult(const DetectionResult& res) {
  boxes.assign(res.boxes.begin(), res.boxes.end());
  scores.assign(res.scores.begin(), res.scores.end());
  label_ids.assign(res.label_ids.begin(), res.label_ids.end());
  contain_masks = res.contain_masks;
  if (contain_masks) {
    masks.clear();
    size_t mask_size = res.masks.size();
    for (size_t i = 0; i < mask_size; ++i) {
      masks.emplace_back(res.masks[i]);
    }
  }
}

void DetectionResult::Clear() {
  std::vector<std::array<float, 4>>().swap(boxes);
  std::vector<float>().swap(scores);
  std::vector<int32_t>().swap(label_ids);
  std::vector<Mask>().swap(masks);
  contain_masks = false;
}

void DetectionResult::Reserve(int size) {
  boxes.reserve(size);
  scores.reserve(size);
  label_ids.reserve(size);
  masks.reserve(size);
}

void DetectionResult::Resize(int size) {
  boxes.resize(size);
  scores.resize(size);
  label_ids.resize(size);
  masks.resize(size);
}

std::string DetectionResult::Str() {
  std::string out;
  if (!contain_masks) {
    out = "DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]\n";
  } else {
    out =
        "DetectionResult: [xmin, ymin, xmax, ymax, score, label_id, "
        "mask_shape]\n";
  }
  for (size_t i = 0; i < boxes.size(); ++i) {
    out = out + std::to_string(boxes[i][0]) + "," +
          std::to_string(boxes[i][1]) + ", " + std::to_string(boxes[i][2]) +
          ", " + std::to_string(boxes[i][3]) + ", " +
          std::to_string(scores[i]) + ", " + std::to_string(label_ids[i]);
    if (!contain_masks) {
      out += "\n";
    } else {
      out += ", " + masks[i].Str();
    }
  }
  return out;
}

void KeyPointDetectionResult::Clear() {
  std::vector<std::array<float, 2>>().swap(keypoints);
  std::vector<float>().swap(scores);
  num_joints = -1;
}

void KeyPointDetectionResult::Reserve(int size) { keypoints.reserve(size); }

void KeyPointDetectionResult::Resize(int size) { keypoints.resize(size); }

std::string KeyPointDetectionResult::Str() {
  std::string out;

  out = "KeyPointDetectionResult: [x, y, conf]\n";
  for (size_t i = 0; i < keypoints.size(); ++i) {
    out = out + std::to_string(keypoints[i][0]) + "," +
          std::to_string(keypoints[i][1]) + ", " +
          std::to_string(scores[i]) + "\n";
  }
  out += "num_joints:" + std::to_string(num_joints) + "\n";
  return out;
}

void OCRResult::Clear() {
  boxes.clear();
  text.clear();
  rec_scores.clear();
  cls_scores.clear();
  cls_labels.clear();
}

void MOTResult::Clear(){
  boxes.clear();
  ids.clear();
  scores.clear();
  class_ids.clear();
}

std::string MOTResult::Str(){
  std::string out;
  out = "MOTResult:\nall boxes counts: "+std::to_string(boxes.size())+"\n";
  out += "[xmin\tymin\txmax\tymax\tid\tscore]\n";
  for (size_t i = 0; i < boxes.size(); ++i) {
    out = out + "["+ std::to_string(boxes[i][0]) + "\t" +
          std::to_string(boxes[i][1]) + "\t" + std::to_string(boxes[i][2]) +
          "\t" + std::to_string(boxes[i][3]) + "\t" +
          std::to_string(ids[i]) + "\t" + std::to_string(scores[i]) + "]\n";
  }
  return out;
}

FaceDetectionResult::FaceDetectionResult(const FaceDetectionResult& res) {
  boxes.assign(res.boxes.begin(), res.boxes.end());
  landmarks.assign(res.landmarks.begin(), res.landmarks.end());
  scores.assign(res.scores.begin(), res.scores.end());
  landmarks_per_face = res.landmarks_per_face;
}

void FaceDetectionResult::Clear() {
  std::vector<std::array<float, 4>>().swap(boxes);
  std::vector<float>().swap(scores);
  std::vector<std::array<float, 2>>().swap(landmarks);
  landmarks_per_face = 0;
}

void FaceDetectionResult::Reserve(int size) {
  boxes.reserve(size);
  scores.reserve(size);
  if (landmarks_per_face > 0) {
    landmarks.reserve(size * landmarks_per_face);
  }
}

void FaceDetectionResult::Resize(int size) {
  boxes.resize(size);
  scores.resize(size);
  if (landmarks_per_face > 0) {
    landmarks.resize(size * landmarks_per_face);
  }
}

std::string FaceDetectionResult::Str() {
  std::string out;
  // format without landmarks
  if (landmarks_per_face <= 0) {
    out = "FaceDetectionResult: [xmin, ymin, xmax, ymax, score]\n";
    for (size_t i = 0; i < boxes.size(); ++i) {
      out = out + std::to_string(boxes[i][0]) + "," +
            std::to_string(boxes[i][1]) + ", " + std::to_string(boxes[i][2]) +
            ", " + std::to_string(boxes[i][3]) + ", " +
            std::to_string(scores[i]) + "\n";
    }
    return out;
  }
  // format with landmarks
  FDASSERT((landmarks.size() == boxes.size() * landmarks_per_face),
           "The size of landmarks != boxes.size * landmarks_per_face.");
  out = "FaceDetectionResult: [xmin, ymin, xmax, ymax, score, (x, y) x " +
        std::to_string(landmarks_per_face) + "]\n";
  for (size_t i = 0; i < boxes.size(); ++i) {
    out = out + std::to_string(boxes[i][0]) + "," +
          std::to_string(boxes[i][1]) + ", " + std::to_string(boxes[i][2]) +
          ", " + std::to_string(boxes[i][3]) + ", " +
          std::to_string(scores[i]) + ", ";
    for (size_t j = 0; j < landmarks_per_face; ++j) {
      out = out + "(" +
            std::to_string(landmarks[i * landmarks_per_face + j][0]) + "," +
            std::to_string(landmarks[i * landmarks_per_face + j][1]);
      if (j < landmarks_per_face - 1) {
        out = out + "), ";
      } else {
        out = out + ")\n";
      }
    }
  }
  return out;
}

void FaceAlignmentResult::Clear() {
  std::vector<std::array<float, 2>>().swap(landmarks);
}

void FaceAlignmentResult::Reserve(int size) {
  landmarks.resize(size);
}

void FaceAlignmentResult::Resize(int size) {
  landmarks.resize(size);
}

std::string FaceAlignmentResult::Str() {
  std::string out;

  out = "FaceAlignmentResult: [x, y]\n";
  out = out + "There are " +std::to_string(landmarks.size()) + " landmarks, the top 10 are listed as below:\n";
  int landmarks_size = landmarks.size();
  size_t result_length = std::min(10, landmarks_size);
  for (size_t i = 0; i < result_length; ++i) {
    out = out + std::to_string(landmarks[i][0]) + "," +
          std::to_string(landmarks[i][1]) + "\n";
  }
  out += "num_landmarks:" + std::to_string(landmarks.size()) + "\n";
  return out;
}

void SegmentationResult::Clear() {
  std::vector<uint8_t>().swap(label_map);
  std::vector<float>().swap(score_map);
  std::vector<int64_t>().swap(shape);
  contain_score_map = false;
}

void SegmentationResult::Reserve(int size) {
  label_map.reserve(size);
  if (contain_score_map > 0) {
    score_map.reserve(size);
  }
}

void SegmentationResult::Resize(int size) {
  label_map.resize(size);
  if (contain_score_map) {
    score_map.resize(size);
  }
}

std::string SegmentationResult::Str() {
  std::string out;
  out = "SegmentationResult Image masks 10 rows x 10 cols: \n";
  for (size_t i = 0; i < 10; ++i) {
    out += "[";
    for (size_t j = 0; j < 10; ++j) {
      out = out + std::to_string(label_map[i * 10 + j]) + ", ";
    }
    out += ".....]\n";
  }
  out += "...........\n";
  if (contain_score_map) {
    out += "SegmentationResult Score map 10 rows x 10 cols: \n";
    for (size_t i = 0; i < 10; ++i) {
      out += "[";
      for (size_t j = 0; j < 10; ++j) {
        out = out + std::to_string(score_map[i * 10 + j]) + ", ";
      }
      out += ".....]\n";
    }
    out += "...........\n";
  }
  out += "result shape is: [" + std::to_string(shape[0]) + " " +
         std::to_string(shape[1]) + "]";
  return out;
}

FaceRecognitionResult::FaceRecognitionResult(const FaceRecognitionResult& res) {
  embedding.assign(res.embedding.begin(), res.embedding.end());
}

void FaceRecognitionResult::Clear() { std::vector<float>().swap(embedding); }

void FaceRecognitionResult::Reserve(int size) { embedding.reserve(size); }

void FaceRecognitionResult::Resize(int size) { embedding.resize(size); }

std::string FaceRecognitionResult::Str() {
  std::string out;
  out = "FaceRecognitionResult: [";
  size_t numel = embedding.size();
  if (numel <= 0) {
    return out + "Empty Result]";
  }
  // max, min, mean
  float min_val = embedding.at(0);
  float max_val = embedding.at(0);
  float total_val = embedding.at(0);
  for (size_t i = 1; i < numel; ++i) {
    float val = embedding.at(i);
    total_val += val;
    if (val < min_val) {
      min_val = val;
    }
    if (val > max_val) {
      max_val = val;
    }
  }
  float mean_val = total_val / static_cast<float>(numel);
  out = out + "Dim(" + std::to_string(numel) + "), " + "Min(" +
        std::to_string(min_val) + "), " + "Max(" + std::to_string(max_val) +
        "), " + "Mean(" + std::to_string(mean_val) + ")]\n";
  return out;
}

MattingResult::MattingResult(const MattingResult& res) {
  alpha.assign(res.alpha.begin(), res.alpha.end());
  foreground.assign(res.foreground.begin(), res.foreground.end());
  shape.assign(res.shape.begin(), res.shape.end());
  contain_foreground = res.contain_foreground;
}

void MattingResult::Clear() {
  std::vector<float>().swap(alpha);
  std::vector<float>().swap(foreground);
  std::vector<int64_t>().swap(shape);
  contain_foreground = false;
}

void MattingResult::Reserve(int size) {
  alpha.reserve(size);
  if (contain_foreground) {
    FDASSERT((shape.size() == 3),
             "Please initial shape (h,w,c) before call Reserve.");
    int c = static_cast<int>(shape[2]);
    foreground.reserve(size * c);
  }
}

void MattingResult::Resize(int size) {
  alpha.resize(size);
  if (contain_foreground) {
    FDASSERT((shape.size() == 3),
             "Please initial shape (h,w,c) before call Resize.");
    int c = static_cast<int>(shape[2]);
    foreground.resize(size * c);
  }
}

std::string MattingResult::Str() {
  std::string out;
  out = "MattingResult[";
  if (contain_foreground) {
    out += "Foreground(true)";
  } else {
    out += "Foreground(false)";
  }
  out += ", Alpha(";
  size_t numel = alpha.size();
  if (numel <= 0) {
    return out + "[Empty Result]";
  }
  // max, min, mean
  float min_val = alpha.at(0);
  float max_val = alpha.at(0);
  float total_val = alpha.at(0);
  for (size_t i = 1; i < numel; ++i) {
    float val = alpha.at(i);
    total_val += val;
    if (val < min_val) {
      min_val = val;
    }
    if (val > max_val) {
      max_val = val;
    }
  }
  float mean_val = total_val / static_cast<float>(numel);
  // shape
  std::string shape_str = "Shape(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if ((i + 1) != shape.size()) {
      shape_str += std::to_string(shape[i]) + ",";
    } else {
      shape_str += std::to_string(shape[i]) + ")";
    }
  }
  out = out + "Numel(" + std::to_string(numel) + "), " + shape_str + ", Min(" +
        std::to_string(min_val) + "), " + "Max(" + std::to_string(max_val) +
        "), " + "Mean(" + std::to_string(mean_val) + "))]\n";
  return out;
}

std::string OCRResult::Str() {
  std::string no_result;
  if (boxes.size() > 0) {
    std::string out;
    for (int n = 0; n < boxes.size(); n++) {
      out = out + "det boxes: [";
      for (int i = 0; i < 4; i++) {
        out = out + "[" + std::to_string(boxes[n][i * 2]) + "," +
              std::to_string(boxes[n][i * 2 + 1]) + "]";

        if (i != 3) {
          out = out + ",";
        }
      }
      out = out + "]";

      if (rec_scores.size() > 0) {
        out = out + "rec text: " + text[n] + " rec score:" +
              std::to_string(rec_scores[n]) + " ";
      }
      if (cls_labels.size() > 0) {
        out = out + "cls label: " + std::to_string(cls_labels[n]) +
              " cls score: " + std::to_string(cls_scores[n]);
      }
      out = out + "\n";
    }
    return out;

  } else if (boxes.size() == 0 && rec_scores.size() > 0 &&
             cls_scores.size() > 0) {
    std::string out;
    for (int i = 0; i < rec_scores.size(); i++) {
      out = out + "rec text: " + text[i] + " rec score:" +
            std::to_string(rec_scores[i]) + " ";
      out = out + "cls label: " + std::to_string(cls_labels[i]) +
            " cls score: " + std::to_string(cls_scores[i]);
      out = out + "\n";
    }
    return out;
  } else if (boxes.size() == 0 && rec_scores.size() == 0 &&
             cls_scores.size() > 0) {
    std::string out;
    for (int i = 0; i < cls_scores.size(); i++) {
      out = out + "cls label: " + std::to_string(cls_labels[i]) +
            " cls score: " + std::to_string(cls_scores[i]);
      out = out + "\n";
    }
    return out;
  } else if (boxes.size() == 0 && rec_scores.size() > 0 &&
             cls_scores.size() == 0) {
    std::string out;
    for (int i = 0; i < rec_scores.size(); i++) {
      out = out + "rec text: " + text[i] + " rec score:" +
            std::to_string(rec_scores[i]) + " ";
      out = out + "\n";
    }
    return out;
  }

  no_result = no_result + "No Results!";
  return no_result;
}

void HeadPoseResult::Clear() {
  std::vector<float>().swap(euler_angles);
}

void HeadPoseResult::Reserve(int size) {
  euler_angles.resize(size);
}

void HeadPoseResult::Resize(int size) {
  euler_angles.resize(size);
}

std::string HeadPoseResult::Str() {
  std::string out;

  out = "HeadPoseResult: [yaw, pitch, roll]\n";
  out = out + "yaw: " + std::to_string(euler_angles[0]) + "\n" +
        "pitch: " + std::to_string(euler_angles[1]) + "\n" +
        "roll: " + std::to_string(euler_angles[2]) + "\n";
  return out;
}


}  // namespace vision
}  // namespace fastdeploy
