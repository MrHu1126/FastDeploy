# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
'''
This module is used to store environmental variables for fastdeploy model hub.

FASTDEPLOY_HUB_HOME              -->  the root directory for storing fastdeploy model hub related data. Default to ~/.fastdeploy. Users can change the
├                          default value through the FASTDEPLOY_HUB_HOME environment variable.
├── MODEL_HOME       -->   Store the downloaded fastdeploy models.
├── CONF_HOME        -->   Store the default configuration files.
'''

import os


def _get_user_home():
    return os.path.expanduser('~')


def _get_hub_home():
    if 'FASTDEPLOY_HUB_HOME' in os.environ:
        home_path = os.environ['FASTDEPLOY_HUB_HOME']
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                raise RuntimeError('The environment variable FASTDEPLOY_HUB_HOME {} is not a directory.'.format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), '.fastdeploy')


def _get_sub_home(directory):
    home = os.path.join(_get_hub_home(), directory)
    os.makedirs(home, exist_ok=True)
    return home


USER_HOME = _get_user_home()
HUB_HOME = _get_hub_home()
MODEL_HOME = _get_sub_home('models')
CONF_HOME = _get_sub_home('conf')
