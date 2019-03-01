"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import io
import os
import xml.etree.ElementTree as ET
from collections import namedtuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util


def split_labels(labels_csv, train_csv, test_csv, split_rate=0.7):
    full_labels = pd.read_csv(labels_csv)
    gb = full_labels.groupby('filename')
    grouped_list = [gb.get_group(x) for x in gb.groups]
    total = len(grouped_list)
    train_index = np.random.choice(len(grouped_list), size=int(total * split_rate), replace=False)
    test_index = np.setdiff1d(list(range(200)), train_index)
    train = pd.concat([grouped_list[i] for i in train_index])
    test = pd.concat([grouped_list[i] for i in test_index])
    train.to_csv(train_csv, index=None)
    test.to_csv(test_csv, index=None)
    return train_csv, test_csv


def xml_to_csv(path, label_csv, categories):
    cat = [c["name"] for c in categories]
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.findall('object') is not None:
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         cat.index(member[0].text) + 1,
                         member[0].text,
                         int(member[4][1].text),
                         int(member[4][0].text),
                         int(member[4][3].text),
                         int(member[4][2].text)
                         )
                xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'name', 'ymin', 'xmin', 'ymax', 'xmax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv(label_csv, index=None)
    print('Successfully converted xml to csv.')


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(row['class'])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def generate_tfrecord(images_path, csv_input, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, images_path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
