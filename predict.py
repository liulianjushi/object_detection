import os

import cv2
import numpy as np
import tensorflow as tf
from imutils import paths
from imutils.video import WebcamVideoStream
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import matplotlib.pyplot as plt

# from utils.boxes_to_xml import generate_xml
from utils.boxes_to_xml import generate_xml

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    try:
        # 设置GPU显存占用为按需分配
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # 异常处理
        print(e)

saved_model = "models/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model"
label_path = "data/rubbish_label_map.pbtxt"


def load_model(saved_model_dir, label_path, num_classes):
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    # predict_fn = predictor_factories.from_saved_model(saved_model_dir)
    predict_fn = tf.saved_model.load(saved_model_dir)
    return predict_fn, category_index


def predict(predictor, image_url):
    image = cv2.cvtColor(cv2.imread(image_url), cv2.COLOR_BGR2RGB)
    # image = cv2.imread(image_url)
    output_dict = predictor(np.expand_dims(image, axis=0))
    # print(output_dict)
    output_dict['detection_classes'] = output_dict['detection_classes'][0].numpy().astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0].numpy()
    output_dict['detection_scores'] = output_dict['detection_scores'][0].numpy()
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


def online_video_detection(predictor, category_index, url):
    vs = WebcamVideoStream(url).start()
    while vs.stream.isOpened():
        frame = vs.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_dict = predictor(np.expand_dims(image, axis=0))
        output_dict['detection_classes'] = output_dict['detection_classes'][0].numpy().astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0].numpy()
        output_dict['detection_scores'] = output_dict['detection_scores'][0].numpy()
        image = darw_box(frame, output_dict, category_index)
        if len(output_dict['detection_boxes'] != 0):
            pass
        key = cv2.waitKey(1) & 0xFF
        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', image)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('frame', image)
        if key == 27:  # exit
            break
    vs.stop()


def offline_video_detection(predictor, category_index, video_path):
    if video_path == "0":
        video_path = int(video_path)
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
            output_dict = predictor(np.expand_dims(image, axis=0))
            output_dict['detection_classes'] = output_dict['detection_classes'][0].numpy().astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0].numpy()
            output_dict['detection_scores'] = output_dict['detection_scores'][0].numpy()
            image = darw_box(frame, output_dict, category_index)
            key = cv2.waitKey(1) & 0xFF
            # keybindings for display
            if key == ord('p'):  # pause
                while True:
                    key2 = cv2.waitKey(1) or 0xff
                    cv2.imshow('frame', image)
                    if key2 == ord('p'):  # resume
                        break
            cv2.imshow('frame', image)
            if key == 27:  # exit
                break
        else:
            break
    cap.release()


def main(predictor, category_index, image_url):
    # new_path = image_url.replace("images1", "xml1")
    # if not os.path.exists(new_path):
    #     os.makedirs(new_path)
    if os.path.isfile(image_url):
        image, output_dict = predict(predictor, image_url)
        image = darw_box(image, output_dict, category_index)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        # generate_xml(image_url, image.shape, output_dict, category_index, new_path)
    else:
        for file_name in paths.list_images(image_url):
            try:
                print(file_name)
                image, output_dict = predict(predictor, file_name)
                image = darw_box(image, output_dict, category_index)
                # plt.imshow(image)
                # plt.show()
                cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                cv2.imwrite(os.path.join("data/output_img", os.path.basename(file_name)),
                            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                new_path = "/Users/james/tensorflow_datasets/rubbish/xml1"
                generate_xml(file_name, image.shape, output_dict, category_index, new_path)
            except Exception as e:
                print(e)
                continue


if __name__ == '__main__':
    image_url = "/Users/james/tensorflow_datasets/rubbish/images1"
    predictor, category_index = load_model(saved_model, label_path, 1)
    main(predictor, category_index, image_url)
    # online_video_detection(predictor, category_index, "rtmp://210.22.126.170:7771/live/mystream")
    # online_video_detection(predictor, category_index, "/Users/james/tensorflow_datasets/rubbish/video/D3_S20201203195501_E20201203200000.mp4")
