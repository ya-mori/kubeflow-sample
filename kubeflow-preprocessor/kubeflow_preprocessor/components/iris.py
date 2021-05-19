import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    origin = pd.read_csv(os.getenv("DATASET_IRIS", "../data/01_inputs/iris.csv"))
    x_train, x_test, y_train, y_test = run(origin)
    write(x_train, x_test, y_train, y_test)


def run(origin: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # cleaning
    # target_category = iris_df["variety"].unique()
    target_category = ['Setosa', 'Versicolor', 'Virginica']
    origin['target'] = [target_category.index(category) for category in origin.variety]
    x = origin.loc[:, ["petal.length", "petal.width"]]
    y = np_utils.to_categorical(origin.target.values)

    # preprocess
    x_train, x_test, y_train, y_test = split_data(x, y)
    x_train_std, x_test_std = scale(x_train, x_test)
    return x_train_std, x_test_std, y_train, y_test


def split_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
    return x_train, x_test, y_train, y_test


def scale(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_std = scaler.transform(x_train)
    x_test_std = scaler.transform(x_test)
    return x_train_std, x_test_std


def write(x_train, x_test, y_train, y_test):
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _float_list_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def serialize(features, target):
        def _serialize(_features, _target):
            # print(_features, _target)
            feature = {
                'petal_length': _float_feature(_features[0]),
                'petal_width': _float_feature(_features[1]),
                'target': _float_list_feature(_target),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()

        tf_string = tf.py_function(
            _serialize,
            (features, target),  # 上記の関数にこれらの引数を渡す
            tf.string  # 戻り値の型は tf.string
        )
        return tf.reshape(tf_string, ())  # 結果はスカラー

    serialized_features_dataset_train = dataset_train.map(serialize)
    serialized_features_dataset_test = dataset_test.map(serialize)

    train_path = os.getenv("TRAIN_TFRECORD_DIR_IRIS", "../data/02_features/iris/train.tfrecord")
    p_path = Path(train_path).parent
    if not p_path.exists():
        p_path.mkdir(parents=True)
    writer = tf.data.experimental.TFRecordWriter(train_path)

    test_path = os.getenv("TEST_TFRECORD_DIR_IRIS", "../data/02_features/iris/test.tfrecord")
    p_path = Path(test_path).parent
    if not p_path.exists():
        p_path.mkdir(parents=True)
    writer.write(serialized_features_dataset_train)
    writer = tf.data.experimental.TFRecordWriter(test_path)
    writer.write(serialized_features_dataset_test)
