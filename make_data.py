import os
from object_detection.utils import label_map_util

from data.generate_tfrecord import generate_tfrecord, xml_to_csv, split_labels

labels_csv = "data/labels.csv"

train_csv = "data/train_labels.csv"
train_record = "data/train.record"

test_csv = "data/val_labels.csv"
test_record = "data/val.record"

label_map_path = "data/label_map.pbtxt"
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=7, use_display_name=True)


def csv_to_record(images, annotations, rate=0.7):
    if not os.path.exists(labels_csv):
        xml_to_csv(annotations, labels_csv, categories)
    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        split_labels(labels_csv, train_csv, test_csv, rate)
    # generate_tfrecord(images, train_csv, train_record)
    # generate_tfrecord(images, test_csv, test_record)


if __name__ == '__main__':
    annotations_path = "/data/zl/南瑞项目素材收集-新版/01.大型机械/01.标注/02.湖北现场"
    images_path = "/data/zl/南瑞项目素材收集-新版/01.大型机械/00.图片/02.湖北现场"
    csv_to_record(images_path, annotations_path, 0.8)
