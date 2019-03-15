import collections
import math

import cv2
import pandas as pd
from object_detection.core import standard_fields
from object_detection.utils import object_detection_evaluation, label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
import os
from tensorflow.contrib.predictor import predictor_factories
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
gpu_options = tf.GPUOptions(allow_growth=True)
tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

label_map_path = "data/label_map.pbtxt"
saved_model = "model/saved_model"

label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=7, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
predict_fn = predictor_factories.from_saved_model(saved_model)

value_dict = collections.OrderedDict([
    ('目标名称', '/'),
    ('图片名称', ''),
    ('数据来源', '湖北现场'),
    ('场景类别', '输电'),
    ('标注类别', "/"),
    ('标注名称', "/"),
    ('ymin', '/'),
    ('xmin', '/'),
    ('ymax', '/'),
    ('xmax', '/'),
    ('d_class', "/"),
    ('d_ymin', '/'),
    ('d_xmin', '/'),
    ('d_ymax', '/'),
    ('d_xmax', '/'),
    ('大类', 0),
    ('小类', 0),
    ('IOU', 0.0),
    ('可视结果', '/')
])

xml_list = []


def decide_overlap(boxes, values):
    iou = []
    for value in values:
        ratios = []
        for box in boxes:
            minx = max(box[1], value[1])
            miny = max(box[0], value[0])

            maxx = min(box[3], value[3])
            maxy = min(box[2], value[2])
            if minx < maxx and miny < maxy:
                area = (maxx - minx) * (maxy - miny)
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                rect2_area = (value[2] - value[0]) * (value[3] - value[1])
                ratio = area / (box_area + rect2_area - area)
                ratios.append(ratio)
            else:
                ratios.append(0.0)
        iou.append(ratios)
    return np.array(iou)


def darw_box(image, ground_truth_dict, output_dict, category_index):
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        use_normalized_coordinates=False,
        line_thickness=3)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        ground_truth_dict['ground_truth_boxes'],
        ground_truth_dict['ground_truth_classes'],
        None,
        category_index,
        use_normalized_coordinates=False,
        line_thickness=3)
    return image


def result(image, selected_value, output_dict, image_dir):
    detected_images_path = image_dir.replace("00.图片", "00.图片_检测到")
    undetected_images_path = image_dir.replace("00.图片", "00.图片_未检测到")
    if not os.path.exists(detected_images_path):
        os.makedirs(detected_images_path)
    if not os.path.exists(undetected_images_path):
        os.makedirs(undetected_images_path)
    ground_truth_dict = {"ground_truth_boxes": selected_value.values[:, 5:9],
                         "ground_truth_classes": selected_value.values[:, 3]}
    darw_box(image, ground_truth_dict, output_dict, category_index)
    if len(output_dict["detection_boxes"]) == 0:
        for index, row in selected_value.iterrows():
            v = value_dict.copy()
            v["图片名称"] = str(row["filename"]).split(".")[0]
            v["目标名称"] = "{}_{}".format(v["图片名称"], str(index).zfill(2))
            v["标注类别"] = row["class"]
            v["标注名称"] = row["name"]
            v["ymin"] = row["ymin"]
            v["xmin"] = row["xmin"]
            v["ymax"] = row["ymax"]
            v["xmax"] = row["xmax"]
            v["可视结果"] = row["filename"]
            print(tuple(v.values()))
            xml_list.append(tuple(v.values()))
        cv2.imwrite(os.path.join(undetected_images_path, selected_value.values[0, 0]), image)
        return
    detected_list = [0] * len(output_dict["detection_boxes"])
    ious = decide_overlap(output_dict["detection_boxes"], selected_value.values[:, 5:9])
    for k, iou in enumerate(ious):
        row = selected_value.iloc[k]
        v = value_dict.copy()
        v["图片名称"] = str(row["filename"]).split(".")[0]
        v["目标名称"] = "{}_{}".format(v["图片名称"], str(k).zfill(2))
        v["标注类别"] = row["class"]
        v["标注名称"] = row["name"]
        v["ymin"] = row["ymin"]
        v["xmin"] = row["xmin"]
        v["ymax"] = row["ymax"]
        v["xmax"] = row["xmax"]
        v["可视结果"] = row["filename"]
        max_iou_id = np.argmax(iou)
        if iou[max_iou_id] == np.max(ious[:, max_iou_id]) and iou[max_iou_id] > 0.5 and detected_list[
            int(max_iou_id)] != 1:
            box = (output_dict["detection_boxes"][max_iou_id]).astype(int)
            v["d_class"] = output_dict["detection_classes"][max_iou_id]
            v["d_ymin"] = box[0]
            v["d_xmin"] = box[1]
            v["d_ymax"] = box[2]
            v["d_xmax"] = box[3]
            v["大类"] = 1
            if v["d_class"] == v["标注类别"]:
                v["小类"] = 1
            v["IOU"] = np.max(iou)
            detected_list[int(max_iou_id)] = 1
        else:
            cv2.imwrite(os.path.join(undetected_images_path, row["filename"]), image)
        print(tuple(v.values()))
        xml_list.append(tuple(v.values()))

    for i, det in enumerate(detected_list):
        if det == 1:
            continue
        v = value_dict.copy()
        v["图片名称"] = str(selected_value.values[0, 0]).split(".")[0]
        box = (output_dict["detection_boxes"][i]).astype(int)
        v["d_class"] = output_dict["detection_classes"][i]
        v["d_ymin"] = box[0]
        v["d_xmin"] = box[1]
        v["d_ymax"] = box[2]
        v["d_xmax"] = box[3]
        v["可视结果"] = selected_value.values[0, 0]
        v["大类"] = 0
        print(tuple(v.values()))
        xml_list.append(tuple(v.values()))
    if not os.path.exists(os.path.join(undetected_images_path, selected_value.values[0, 0])):
        cv2.imwrite(os.path.join(detected_images_path, selected_value.values[0, 0]), image)
    return


def read_csv(image_dir, labels_csv):
    power_evaluator = object_detection_evaluation.ObjectDetectionEvaluator(categories=categories,
                                                                           evaluate_precision_recall=True)
    full_labels = pd.read_csv(labels_csv)
    filename = None
    for index, row in full_labels.iterrows():
        if filename == row["filename"]:
            continue
        else:
            filename = row["filename"]
        image = cv2.imread('{}/{}'.format(image_dir, filename))
        selected_value = full_labels[full_labels.filename == filename]
        size_x = image.shape[1]
        size_y = image.shape[0]
        output_dict = predict_fn({"inputs": np.expand_dims(image, axis=0)})
        index = np.where(output_dict['detection_scores'][0] > 0.5)
        output_dict['detection_classes'] = output_dict['detection_classes'][0][index].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0][index] * [size_y, size_x, size_y, size_x]
        output_dict['detection_scores'] = output_dict['detection_scores'][0][index]

        power_evaluator.add_single_ground_truth_image_info(filename, {
            standard_fields.InputDataFields.groundtruth_boxes: np.array(selected_value.values[:, 5:9], dtype=float),
            standard_fields.InputDataFields.groundtruth_classes: np.array(selected_value.values[:, 3], dtype=int)
        })

        power_evaluator.add_single_detected_image_info(filename, {
            standard_fields.DetectionResultFields.detection_boxes: output_dict['detection_boxes'],
            standard_fields.DetectionResultFields.detection_scores: output_dict['detection_scores'],
            standard_fields.DetectionResultFields.detection_classes: output_dict['detection_classes'],
        })
        result(image, selected_value, output_dict, image_dir)

    metrics = power_evaluator.evaluate()
    xml_list.append([""])
    xml_list.append([""])
    xml_list.append(['category_name', 'ap', 'precision', 'recall'])
    for c in categories:
        c_list = [c["name"]]
        precision = metrics["PerformanceByCategory/Precision@0.5IOU/b'{}'".format(c["name"])]
        ap = metrics["PerformanceByCategory/AP@0.5IOU/b'{}'".format(c["name"])]
        recall = metrics["PerformanceByCategory/Recall@0.5IOU/b'{}'".format(c["name"])]
        if math.isnan(ap):
            c_list.append("nan")
        else:
            c_list.append(ap)

        if isinstance(precision, float):
            c_list.append("nan")
        else:
            c_list.append(precision[-1])

        if isinstance(recall, float):
            c_list.append("nan")
        else:
            c_list.append(recall[-1])
        xml_list.append(c_list)
    xml_list.append(['Precision/mAP@0.5IOU', metrics['Precision/mAP@0.5IOU']])


if __name__ == '__main__':
    read_csv("/data/zl/南瑞项目素材收集-新版/01.大型机械/00.图片/02.湖北现场", "data/labels.csv")
    xml_df = pd.DataFrame(xml_list, columns=list(value_dict.keys()))
    xml_df.to_csv('big_mechanics_test.csv', index=None)
