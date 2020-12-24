import json
import cv2
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detector import DetectorTF2

saved_model = "models/my_efficientdet_d0/exported_model/saved_model"
label_path = "data/label_map.pbtxt"
detector = DetectorTF2(saved_model, label_path)
print(detector.category_index)


def data_eval(images_dir, json_file, save_file):
    with open(json_file) as f:
        data = json.load(f)
    dataset = []
    for item in data["images"]:
        image = cv2.cvtColor(cv2.imread(os.path.join(images_dir, item["file_name"])), cv2.COLOR_BGR2RGB)
        boxes = detector.DetectFromImage(image)
        for box in boxes:
            ## [x_min, y_min, x_max, y_max, class_label, float(bscores[idx])]
            dataset.append({"image_id": item["id"], "category_id": box[4],
                            "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]], "score": box[6]})
            print({"image_id": item["id"], "category_id": box[4],
                   "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]], "score": box[6]})

    with open(save_file, 'w') as f:
        json.dump(dataset, f)


if __name__ == '__main__':
    # data_eval("/Users/james/tensorflow_datasets/coco/JPEGImages",
    #           '/Users/james/tensorflow_datasets/coco/json2014/annotations/instances_val2014.json',
    #           "data/val_labels_result.json")

    cocoGt = COCO("/Users/james/tensorflow_datasets/coco/json2014/annotations/instances_val2014.json")
    cocoDt = cocoGt.loadRes("data/val_labels_result.json")
    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
