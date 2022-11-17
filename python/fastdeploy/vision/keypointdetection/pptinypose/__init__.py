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

from __future__ import absolute_import
import logging
from .... import FastDeployModel, ModelFormat
from .... import c_lib_wrap as C


class PPTinyPose(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """load a PPTinyPose model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g pptinypose/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g pptinypose/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g pptinypose/infer_cfg.yml
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """
        super(PPTinyPose, self).__init__(runtime_option)

        assert model_format == ModelFormat.PADDLE, "PPTinyPose model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.keypointdetection.PPTinyPose(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PPTinyPose model initialize failed."

    def predict(self, input_image):
        """Detect keypoints in an input image

        :param im: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: KeyPointDetectionResult
        """
        assert input_image is not None, "The input image data is None."
        return self._model.predict(input_image)

    def predict_with_detection_result(self, input_image, detection_result):
        """Detect keypoints in an input image with detection result

        :param im: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :param im: Fastdeploy DetectionResult
        :return: KeyPointDetectionResult
        """
        assert input_image is not None, "The input image data is None."
        assert detection_result is not None, "The detection_result data is None"
        return self._model.predict(input_image, detection_result)

    @property
    def use_dark(self):
        """Atrribute of PPTinyPose model. Stating whether using Distribution-Aware Coordinate Representation for Human Pose Estimation(DARK for short) in postprocess, default is True

        :return: value of use_dark(bool)
        """
        return self._model.use_dark

    @use_dark.setter
    def use_dark(self, value):
        """Set attribute use_dark of PPTinyPose model.

        :param value: (bool)The value to set use_dark
        """
        assert isinstance(
            value, bool), "The value to set `use_dark` must be type of bool."
        self._model.use_dark = value
