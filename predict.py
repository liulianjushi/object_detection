import os

import cv2
import numpy as np
import tensorflow as tf
from imutils import paths
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from tensorflow.contrib.predictor import predictor_factories
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
gpu_options = tf.GPUOptions(allow_growth=True)
tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

saved_model = "model/saved_model"
label_path = "data/label_map.pbtxt"


def load_model(saved_model_dir, label_path, num_classes):
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    predict_fn = predictor_factories.from_saved_model(saved_model_dir)
    return predict_fn, category_index


def predict(predictor, image_url):
    image = cv2.cvtColor(cv2.imread(image_url), cv2.COLOR_BGR2RGB)
    output_dict = predictor({"inputs": np.expand_dims(image, axis=0)})
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return image, output_dict


def darw_box(image, output_dict, category_index):
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3)
    return image


def main(predictor, category_index, image_url):
    new_path = image_url.replace("00.图片", "00.图片_检测结果")
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    if os.path.isfile(image_url):
        image, output_dict = predict(predictor, image_url)
        image = darw_box(image, output_dict, category_index)
        cv2.imwrite(os.path.join(new_path, image_url.split("/")[-1]), image)
    else:
        for file_name in tqdm(paths.list_images(image_url)):
            try:
                image, output_dict = predict(predictor, file_name)
                image = darw_box(image, output_dict, category_index)
                cv2.imwrite(os.path.join(new_path, file_name.split("/")[-1]), image)
            except Exception as e:
                continue


if __name__ == '__main__':
    image_url = "../南瑞项目素材收集-新版/01.大型机械/00.图片/02.湖北现场"
    predictor, category_index = load_model(saved_model, label_path, 2)
    main(predictor, category_index, image_url)
