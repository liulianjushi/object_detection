import glob
import xml.etree.ElementTree as ET

import cv2
import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import matplotlib.pyplot as plt


def read_xml_annotation(image_dir, xml_dir):
    for xml_file in glob.glob(xml_dir + '/*.xml')[:20]:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        image = cv2.imread('{}/{}'.format(image_dir, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = []

        if root.findall('object') is not None:
            for member in root.findall('object'):
                boxes.append(BoundingBox(int(member[4][0].text), int(member[4][1].text),
                                         int(member[4][2].text), int(member[4][3].text)))
        bbs = BoundingBoxesOnImage(boxes, shape=image.shape)

        seq = iaa.Sequential([
            iaa.Flipud(0.5),  # v翻转
            iaa.Fliplr(0.5),  # 镜像
            iaa.Multiply((1.2, 1.5)),  # 改变明亮度
            iaa.GaussianBlur(sigma=(0, 3.0)),  # 高斯噪声
            iaa.Affine(translate_px={"x": 15, "y": 15}, scale=(0.8, 0.95), rotate=(-30, 30))
            # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
        ])

        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        if root.findall('object') is not None:
            for index, member in enumerate(root.findall('object')):
                after = bbs_aug.bounding_boxes[index]
                bndbox = member.find('bndbox')  # 子节点下节点rank的值

                xmin = bndbox.find('xmin')
                xmin.text = str(after.x1)
                ymin = bndbox.find('ymin')
                ymin.text = str(after.y1)
                xmax = bndbox.find('xmax')
                xmax.text = str(after.x2)
                ymax = bndbox.find('ymax')
                ymax.text = str(after.y2)

        cv2

        # image with BBs before/after augmentation (shown below)
        # image_before = bbs.draw_on_image(image, size=2)
        # plt.imshow(image_before)
        # plt.show()
        image_after = bbs_aug.draw_on_image(image_aug, size=4, color=[0, 0, 255])
        plt.imshow(image_after)
        plt.show()


if __name__ == '__main__':
    xml_dir = "/Users/james/Documents/haircut-zip/xml"
    images_dir = "/Users/james/Documents/haircut-zip/images"
    read_xml_annotation(images_dir, xml_dir)
