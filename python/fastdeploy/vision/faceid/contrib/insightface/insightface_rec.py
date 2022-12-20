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
from python.fastdeploy import FastDeployModel, ModelFormat
from python.fastdeploy import c_lib_wrap as C


class InsightFaceRecognitionBase(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file="",
                 runtime_option=None,
                 model_format=ModelFormat.ONNX):
        """Load a InsightFace model exported by InsigtFace.

        :param model_file: (str)Path of model file, e.g ./arcface.onnx
        :param params_file: (str)Path of parameters file, e.g yolox/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """
        # 调用基函数进行backend_option的初始化
        # 初始化后的option保存在self._runtime_option
        super(InsightFaceRecognitionBase, self).__init__(runtime_option)

        self._model = C.vision.faceid.InsightFaceRecognitionBase(
            model_file, params_file, self._runtime_option, model_format)
        # 通过self.initialized判断整个模型的初始化是否成功
        assert self.initialized, "InsightFaceRecognitionModel initialize failed."

    def predict(self, input_image):
        """ Predict the face recognition result for an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: FaceRecognitionResult
        """
        return self._model.predict(input_image)
