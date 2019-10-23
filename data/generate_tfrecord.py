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
from object_detection.utils import dataset_util, label_map_util


class GenerateTFRecord(object):
    def __init__(self, label_map_path=None, max_num_classes=2):
        self.labels_csv = "data/labels.csv"
        self.train_csv = "data/train_labels.csv"
        self.train_record = "data/train.record"
        self.test_csv = "data/val_labels.csv"
        self.test_record = "data/val.record"
        self.label_map_path = label_map_path
        self.label_map = label_map_util.load_labelmap(self.label_map_path)
        self.max_num_classes = max_num_classes
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                         max_num_classes=self.max_num_classes,
                                                                         use_display_name=True)

    def split(self, df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

    def create_tf_example(self, group, path):
        with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
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
            classes_text.append(row['name'].encode('utf8'))
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

    def xml_to_csv(self, path):
        cat = [c["name"] for c in self.categories]
        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            print(xml_file)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            if root.findall('object') is not None:
                for member in root.findall('object'):
                    xmin = int(member[4][0].text)
                    ymin = int(member[4][1].text)
                    xmax = int(member[4][2].text)
                    ymax = int(member[4][3].text)

                    width = int(root.find('size')[0].text)
                    height = int(root.find('size')[1].text)

                    if xmin > width or ymin > height or xmax < 0 or ymax < 0:
                        continue

                    if xmin < 0:
                        xmin = 0
                    if ymin < 0:
                        ymin = 0
                    if xmax > width:
                        xmax = width
                    if ymax > height:
                        ymax = height

                    value = (
                    root.find('filename').text, width, height, cat.index(member[0].text) + 1, member[0].text, ymin,
                    xmin, ymax, xmax)
                    xml_list.append(value)
        column_name = ['filename', 'width', 'height', 'class', 'name', 'ymin', 'xmin', 'ymax', 'xmax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        xml_df.to_csv(self.labels_csv, index=None)
        print('Successfully converted xml to csv.')

    def split_labels(self, split_rate=0.7):
        full_labels = pd.read_csv(self.labels_csv)
        gb = full_labels.groupby('filename')
        grouped_list = [gb.get_group(x) for x in gb.groups]
        total = len(grouped_list)
        train_index = np.random.choice(len(grouped_list), size=int(total * split_rate), replace=False)
        test_index = np.setdiff1d(list(range(total)), train_index)
        train = pd.concat([grouped_list[i] for i in train_index])
        test = pd.concat([grouped_list[i] for i in test_index])
        train_num = train.groupby("name").size().reset_index(name='number')
        val_num = test.groupby("name").size().reset_index(name='number')
        train.to_csv(self.train_csv, index=None)
        test.to_csv(self.test_csv, index=None)
        print("Number of train dataset is:\n ", train_num)
        print("Number of val dataset is:\n", val_num)

    def generate_tfrecord(self, images_path, csv_input, output_path):
        writer = tf.io.TFRecordWriter(output_path)
        examples = pd.read_csv(csv_input)
        grouped = self.split(examples, 'filename')
        for group in grouped:
            tf_example = self.create_tf_example(group, images_path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the TFRecords: {}'.format(output_path))

    def make_data(self, images_path, annotations_path, rate):
        if not os.path.exists(self.labels_csv):
            self.xml_to_csv(annotations_path)
        # if not (os.path.exists(self.train_csv) and os.path.exists(self.test_csv)):
        #     self.split_labels(rate)
        # self.generate_tfrecord(images_path, self.train_csv, self.train_record)
        # self.generate_tfrecord(images_path, self.test_csv, self.test_record)


if __name__ == '__main__':
    annotations_path = "/Users/james/Downloads/xml1"
    images_path = "/Users/james/Documents/haircut/images1"
    label_map_path = "data/label_map.pbtxt"

    generate_data = GenerateTFRecord(label_map_path=label_map_path, max_num_classes=2)
    generate_data.make_data(images_path, annotations_path, 0.7)
