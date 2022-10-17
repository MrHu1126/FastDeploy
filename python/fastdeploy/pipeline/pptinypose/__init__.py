# # Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

from __future__ import absolute_import
from ... import c_lib_wrap as C


class PPTinyPose(object):
    def __init__(self, det_model=None, pptinypose_model=None):
        assert det_model is not None and pptinypose_model is not None, "The det_model and pptinypose_model cannot be None."
        self._pipeline = C.pipeline.PPTinyPose(det_model._model,
                                               pptinypose_model._model)

    def predict(self, input_image):
        return self._pipeline.predict(input_image)

    @property
    def detect_model_score_threshold(self):
        return self._model.detect_model_score_threshold

    @detect_model_score_threshold.setter
    def detect_model_score_threshold(self, value):
        assert isinstance(
            value, float
        ), "The value to set `detect_model_score_threshold` must be type of float."
        self._pipeline.detect_model_score_threshold = value
