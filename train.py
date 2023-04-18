# Copyright 2019 Bisonai Authors. All Rights Reserved.
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
# ==============================================================================
"""Implementation of paper Searching for MobileNetV3, https://arxiv.org/abs/1905.02244

Training script
"""

import tensorflow as tf
import datetime, os

from mobilenetv3_factory import build_mobilenetv3
from datasets import build_dataset

_available_optimizers = {
    "rmsprop": tf.train.RMSPropOptimizer,
    "adam": tf.train.AdamOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
    }

# 匯入資料集
train_ds, val_ds = build_dataset(
    train_path = "/constent/Dataset/inagenet-mini-160/train/",
    val_path = "/constent/Dataset/inagenet-mini-160/val/"
    )

# 設定基本參數
learning_rate = 0.045
width_multiplier = 0.35

model = build_mobilenetv3(
    model_type = "small",
    input_shape = train_ds.element_spec[0].shape[1:],
    num_classes = len(train_ds.class_names),
    width_multiplier = width_multiplier,
)

if args.optimizer not in _available_optimizers:
    raise NotImplementedError

model.compile(
    optimizer = tf.train.RMSPropOptimizer(learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=logdir),
]

model.fit(
    train_ds,
    validation_data = val_ds,
    #steps_per_epoch=(dataset["num_train"]//args.train_batch_size)+1,
    epochs=60,
    #validation_steps=(dataset["num_test"]//args.valid_batch_size)+1,
    callbacks=callbacks,
)

