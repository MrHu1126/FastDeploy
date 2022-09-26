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

import os
import sys
import numpy as np
import time
import argparse
from tqdm import tqdm
import paddle
from paddleslim.common import load_config, load_onnx_model
from paddleslim.auto_compression import AutoCompression
from paddleslim.quant import quant_post_static
from fdquant.dataset import yolo_image_preprocess, cls_image_preprocess


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--method',
        type=str,
        default=None,
        help="choose PTQ or QAT as quantization method",
        required=True)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
        help="directory to save compressed model.")
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")

    return parser


def reader_wrapper(reader, input_name='x2paddle_images'):
    def gen():
        for data in reader:
            yield {input_name: data[0]}

    return gen


def main():

    time_s = time.time()

    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    assert FLAGS.devices in ['cpu', 'gpu', 'xpu', 'npu']
    paddle.set_device(FLAGS.devices)

    global global_config
    all_config = load_config(FLAGS.config_path)
    assert "Global" in all_config, f"Key 'Global' not found in config file. \n{all_config}"
    global_config = all_config["Global"]
    input_name = global_config['input_name']

    assert os.path.exists(global_config[
        'image_path']), "image_path does not exist!"
    paddle.vision.image.set_image_backend('cv2')
    #transform could be customized.
    train_dataset = paddle.vision.datasets.ImageFolder(
        global_config['image_path'],
        transform=eval(global_config['preprocess']))
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0)
    train_loader = reader_wrapper(train_loader, input_name=input_name)
    eval_func = None

    # ACT compression
    if FLAGS.method == 'QAT':
        ac = AutoCompression(
            model_dir=global_config['model_dir'],
            model_filename=global_config['model_filename'],
            params_filename=global_config['params_filename'],
            train_dataloader=train_loader,
            save_dir=FLAGS.save_dir,
            config=all_config,
            eval_callback=eval_func)
        ac.compress()

    # PTQ compression
    if FLAGS.method == 'PTQ':

        # Read PTQ config
        assert "PTQ" in all_config, f"Key 'PTQ' not found in config file. \n{all_config}"
        ptq_config = all_config["PTQ"]

        # Inititalize the executor
        place = paddle.CUDAPlace(
            0) if FLAGS.devices == 'gpu' else paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        # Read ONNX or PADDLE format model
        if global_config['format'] == 'onnx':
            load_onnx_model(global_config["model_dir"])
            inference_model_path = global_config["model_dir"].rstrip().rstrip(
                '.onnx') + '_infer'
        else:
            inference_model_path = global_config["model_dir"].rstrip('/')

        quant_post_static(
            executor=exe,
            model_dir=inference_model_path,
            quantize_model_path=FLAGS.save_dir,
            data_loader=train_loader,
            model_filename=global_config["model_filename"],
            params_filename=global_config["params_filename"],
            batch_size=32,
            batch_nums=10,
            algo=ptq_config['calibration_method'],
            hist_percent=0.999,
            is_full_quantize=False,
            bias_correction=False,
            onnx_format=True,
            skip_tensor_list=ptq_config['skip_tensor_list']
            if 'skip_tensor_list' in ptq_config else None)

    time_total = time.time() - time_s
    print("Finish Compression, total time used is : ", time_total, "seconds.")


if __name__ == '__main__':
    main()
