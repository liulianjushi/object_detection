import json

import pandas as pd
from collections import namedtuple


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def convert(csv_file, json_name):
    dataset = {'categories': [], 'images': [], 'annotations': []}
    examples = pd.read_csv(csv_file)
    grouped = split(examples, 'filename')
    for image_id, group in enumerate(grouped):
        for index, row in group.object.iterrows():
            dataset['images'].append({'file_name': group.filename,
                                      'id': image_id,
                                      'width': row.width,
                                      'height': row.height})
            if {'id': row["class"], 'name': row["name"], 'supercategory': 'beverage'} not in dataset['categories']:
                dataset['categories'].append({'id': row["class"], 'name': row["name"], 'supercategory': 'beverage'})
                # print({'id': row[3], 'name': row["name"], 'supercategory': 'beverage'})

            dataset['annotations'].append({
                'area': row.width * row.height,
                'bbox': [row.xmin, row.ymin, row.width, row.height],
                'category_id': int(row["class"]),
                'id': row.name,
                'image_id': image_id,
                'iscrowd': 0,
                'segmentation': [[row.xmin, row.ymin, row.xmax, row.ymin, row.xmax, row.ymax, row.xmin, row.ymax]]
            })
            print({
                'area': row.width * row.height,
                'bbox': [row.xmin, row.ymin, row.width, row.height],
                'category_id': int(row["class"]),
                'id': row.name,
                'image_id': image_id,
                'iscrowd': 0,
                'segmentation': [[row.xmin, row.ymin, row.xmax, row.ymin, row.xmax, row.ymax, row.xmin, row.ymax]]
            })
    with open(json_name, 'w') as f:
        json.dump(dataset, f)


if __name__ == '__main__':
    convert("/Users/james/IdeaProjects/object_detection/data/val_labels.csv",
            "/Users/james/IdeaProjects/object_detection/data/val_labels.json")
