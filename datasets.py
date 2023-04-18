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
import tensorflow as tf
from keras.layers.preprocessing.image_preprocessing import Rescaling

def build_dataset(
    train_path: str,
    val_path: str,
    interpolation: str="bilinear",
    size: int=160
    ):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        interpolation=interpolation,
        image_size=(size,size),
        seed=123)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_path,
        interpolation=interpolation,
        image_size=(size,size),
        seed=123)
    
    rescale = Rescaling(1./127.5, offset=-1)
    train_ds = train_ds.map(lambda x, y:(rescale(x), y))
    val_ds = val_ds.map(lambda x, y:(rescale(x), y))

    return train_ds, val_ds
