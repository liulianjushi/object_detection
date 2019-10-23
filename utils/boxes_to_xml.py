import os
import xml.etree.cElementTree as ET

import keras.applications
import numpy as np


def generate_xml(filename, image_shape, output_dict, category_index, xml_dir):
    filename = os.path.basename(filename)

    root = ET.Element('annotation')

    ET.SubElement(root, 'folder').text = 'images'  # set correct folder name
    ET.SubElement(root, 'filename').text = filename

    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = "Unknown"

    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(image_shape[1])
    ET.SubElement(size, 'height').text = str(image_shape[0])
    ET.SubElement(size, 'depth').text = str(image_shape[2])

    ET.SubElement(root, 'segmented').text = '0'

    index = np.where(output_dict['detection_scores'] > 0.5)

    if len(index[0]) != 0:
        output_dict['detection_classes'] = (output_dict['detection_classes'][index])
        output_dict['detection_boxes'] = output_dict['detection_boxes'][index] * [image_shape[0], image_shape[1],
                                                                                  image_shape[0], image_shape[1]]
        output_dict['detection_scores'] = output_dict['detection_scores'][index]
        for box, cl in zip(output_dict["detection_boxes"], output_dict["detection_classes"]):
            name = category_index[cl]["name"]  # class name
            xmin = int(box[1])  # set correct index
            ymin = int(box[0])  # set correct index
            xmax = int(box[3])  # set correct index
            ymax = int(box[2])  # set correct index

            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = name
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'

            bx = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bx, 'xmin').text = str(xmin)
            ET.SubElement(bx, 'ymin').text = str(ymin)
            ET.SubElement(bx, 'xmax').text = str(xmax)
            ET.SubElement(bx, 'ymax').text = str(ymax)

    tree = ET.ElementTree(root)
    tree.write(os.path.join(xml_dir, filename.replace("png", "xml")))


def ground_truth_xml(filename, image_shape, ground_truth_dict, category_index, xml_dir):
    filename = os.path.basename(filename)

    root = ET.Element('annotation')

    ET.SubElement(root, 'folder').text = 'images'  # set correct folder name
    ET.SubElement(root, 'filename').text = filename

    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = "Unknown"

    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(image_shape[1])
    ET.SubElement(size, 'height').text = str(image_shape[0])
    ET.SubElement(size, 'depth').text = str(image_shape[2])

    ET.SubElement(root, 'segmented').text = '0'
    for box, cl in zip(ground_truth_dict["ground_truth_boxes"], ground_truth_dict["ground_truth_classes"]):
        name = category_index[cl]["name"]  # class name
        xmin = int(box[1])  # set correct index
        ymin = int(box[0])  # set correct index
        xmax = int(box[3])  # set correct index
        ymax = int(box[2])  # set correct index

        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = name
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bx = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bx, 'xmin').text = str(xmin)
        ET.SubElement(bx, 'ymin').text = str(ymin)
        ET.SubElement(bx, 'xmax').text = str(xmax)
        ET.SubElement(bx, 'ymax').text = str(ymax)

    tree = ET.ElementTree(root)
    tree.write(os.path.join(xml_dir, filename.replace("png", "xml")))
