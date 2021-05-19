import tensorflow as tf


def main():
    train_dataset = load('../data/02_features/iris/train.tfrecord')
    test_dataset = load('../data/02_features/iris/test.tfrecord')

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(1, 1)),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax),
    ])

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    model.fit(
        train_dataset,
        epochs=1,
        #     validation_data=validation_dataset,
        #     callbacks=callbacks,
    )
    print(model.evaluate(test_dataset))
    model.save(filepath="../data/03_models/iris.h5", save_format="h5")


def load(path: str) -> tf.data.TFRecordDataset:
    raw_dataset = tf.data.TFRecordDataset(path)
    feature_description = {
        'petal_length': tf.io.FixedLenFeature([], tf.float32, default_value=0),
        'petal_width': tf.io.FixedLenFeature([], tf.float32, default_value=0),
        'target': tf.io.FixedLenFeature([3], tf.float32, default_value=[0, 0, 0]),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    def _feature_convert_function(feature):
        return [feature["petal_length"], feature["petal_width"]], feature["target"]

    parsed_dataset = (
        raw_dataset
        .map(_parse_function)
        .map(_feature_convert_function)
    )
    return parsed_dataset
