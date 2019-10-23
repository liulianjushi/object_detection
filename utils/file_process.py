import glob
import os
import shutil
import time

import imutils
import numpy as np
import xml.etree.ElementTree as ET

import cv2


def remove_node(path):
    for xml_file in glob.glob(path + '/*.xml'):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.findall('object') is not None:
            for index, member in enumerate(root.findall('object')):
                # if int(member[4][0].text) == 899 and int(member[4][1].text) == 310:
                #     root.remove(root.findall('object')[index])
                #     continue
                # if int(member[4][0].text) == 975 and int(member[4][1].text) == 280:
                #     root.remove(root.findall('object')[index])
                #     continue
                if int(member[4][0].text) == 501 and int(member[4][1].text) == 269:
                    member[0].text = "largetrunck"
                    continue
                if int(member[4][0].text) == 715 and int(member[4][1].text) == 118:
                    member[0].text = "largetrunck"
                    continue

        tree.write(xml_file)


def copy_file(src1):
    dst = "/data/zl/object_detection/images/"
    src = "/data/zl/南瑞项目素材收集-新版/01.大型机械/00.图片/02.湖北现场"
    files = os.listdir(src1)
    for file in files:
        print(file)
        shutil.copy(os.path.join(src, file), dst)


def video_to_images(video_dir, image_dir, prefix="images", start=0, times=1.0):
    for file in os.listdir(video_dir):
        cap = cv2.VideoCapture(os.path.join(video_dir, file))
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            if frame_num % int(cap.get(cv2.CAP_PROP_FPS) / times) == 0:
                # dst_im = cv2.flip(frame, 0)  #原型：cv2.flip(src, flipCode[, dst]) → dst  flipCode表示对称轴 0：x轴  1：y轴.  -1：both
                # frame = cv2.transpose(dst_im)
                cv2.imwrite(os.path.join(image_dir, "{}_{}.png".format(prefix, str(start).zfill(5))), frame)
                start += 1
        cap.release()


if __name__ == '__main__':
    video_to_images(r"/Users/james/Documents/haircut/video2", r"/Users/james/Documents/haircut/images2",
                    prefix="haircut", start=4629, times=3.0)
    # copy_file("/data/zl/南瑞项目素材收集-新版/01.大型机械/00.图片_未检测到/02.湖北现场")
    # remove_node(r"F:\南瑞项目素材收集-新版\01.大型机械\01.标注\02.湖北现场")
