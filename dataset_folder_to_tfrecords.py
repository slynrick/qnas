import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image


def _get_image_filenames_and_labels(directory):
    filenames = []
    labels = []

    for subdir, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.jpg'):
                label = int(subdir.split('/')[-1])

                filename = os.path.join(subdir, filename)
                filenames.append(filename)
                labels.append(label)

    return filenames, labels

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _process_image(filename, label):
    # Read the image file.
    image = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)

    # Resize the image to the desired input size.
    image = tf.image.resize(image, [224, 224])

    # Normalize the pixel values to be between 0 and 1.
    image = tf.cast(image, tf.float32) / 255.0

    # Create a one-hot encoded label.
    label = tf.one_hot(label, 10)

    return image, label

def create_tfrecord(directory, output_filename):
    writer = tf.io.TFRecordWriter(output_filename)

    for filename, label in _get_image_filenames_and_labels(directory):
        image, label = _process_image(filename, label)

        image = (np.maximum(image, 0) / image.max()) * 255.0
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        img_raw = image.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(224),
        'width': _int64_feature(224),
        'depth': _int64_feature(3),
        "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))

        writer.write(example.SerializeToString())

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, required=True, help='The directory containing the images.')
    parser.add_argument('--output', type=str, required=True, help='The output TFRecord file.')

    args = parser.parse_args()

    create_tfrecord(args.directory, args.output)