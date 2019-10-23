import glob
import os
import xml.etree.ElementTree as ET

import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def read_xml_annotation(image_dir, xml_dir):
    for xml_file in glob.glob(xml_dir + '/*.xml')[:5]:
        print(xml_file)
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

        for i in range(6):

            seq = iaa.SomeOf(3, [
                iaa.Fliplr(p=0.5),  # 水平0.5的概率翻转
                iaa.Flipud(p=1),  ##垂直0.5的概率翻转
                iaa.Crop(percent=(0, 0.3), keep_size=True),  # 0-0.1的数值，分别乘以图片的宽和高为剪裁的像素个数，保持原尺寸
                iaa.Sometimes(p=1,
                              then_list=[iaa.GaussianBlur(sigma=(0, 0.5))],
                              else_list=[iaa.Flipud(p=0.5), iaa.Flipud(p=0.5)]
                              ),  ######以p的概率执行then_list的增强方法，以1-p的概率执行else_list的增强方法，其中then_list,else_list默认为None
                iaa.ContrastNormalization((0.75, 1.5), per_channel=True),
                ####0.75-1.5随机数值为alpha，对图像进行对比度增强，该alpha应用于每个通道
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.3 * 255), per_channel=0.5),
                #### loc 噪声均值，scale噪声方差，50%的概率，对图片进行添加白噪声并应用于每个通道
                iaa.Multiply((0.8, 1.2), per_channel=0.2),  ####20%的图片像素值乘以0.8-1.2中间的数值,用以增加图片明亮度或改变颜色
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-25, 25),
                    shear=(-8, 8))
                ########对图片进行仿射变化，缩放x,y取值范围均为0.8-1.2之间随机值，左右上下移动的值为-0.2-0.2乘以宽高后的随机值
                ########旋转角度为-25到25的随机值，shear剪切取值范围0-360，-8到8随机值进行图片剪切
            ], deterministic=True)

            # 固定变换
            seq_det = seq.to_deterministic()

            # 变换图像和bounding box
            image_aug = seq_det.augment_images([image])[0]
            bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
            if root.findall('object') is not None:
                for index, member in enumerate(root.findall('object')):
                    after = bbs_aug.bounding_boxes[index]
                    bndbox = member.find('bndbox')  # 子节点下节点rank的值

                    xmin = bndbox.find('xmin')
                    xmin.text = str(int(after.x1))
                    ymin = bndbox.find('ymin')
                    ymin.text = str(int(after.y1))
                    xmax = bndbox.find('xmax')
                    xmax.text = str(int(after.x2))
                    ymax = bndbox.find('ymax')
                    ymax.text = str(int(after.y2))

            image_aug = cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR)
            new_filename = xml_file.split(".")[0].split("/")[-1]
            root.find('filename').text = "aug_{}_{}.png".format(new_filename, i + 1)

            cv2.imwrite(os.path.join("", "aug_{}_{}.png".format(new_filename, i+1)), image_aug)
            tree.write(os.path.join("", "aug_{}_{}.xml".format(new_filename, i+1)))


#         image_after = bbs_aug.draw_on_image(image_aug, size=4, color=[0, 0, 255])
#         plt.imshow(image_after)
#         plt.show()


if __name__ == '__main__':
    xml_dir = "/Users/james/Documents/haircut-zip/xml"
    images_dir = "/Users/james/Documents/haircut-zip/images"
    read_xml_annotation(images_dir, xml_dir)
