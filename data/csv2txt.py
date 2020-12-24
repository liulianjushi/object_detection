import json

import pandas as pd
from collections import namedtuple


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def convert(csv_file, json_name):
    examples = pd.read_csv(csv_file)
    grouped = split(examples, 'filename')
    for image_id, group in enumerate(grouped):
        out_file = open('/Users/james/IdeaProjects/object_detection/data/labels/%s.txt' % (group.filename.split(".")[0]), 'w')
        for index, row in group.object.iterrows():
            value = [row["class"], row.xmin, row.ymin, row.xmax, row.ymax]
            print(" ".join([str(a) for a in value]))
            out_file.write(" ".join([str(a) for a in value])+"\n")
        #     labels_txt += " ".join([str(a) for a in value])+" "
        # print(group.filename.split(".")[0]+".txt",labels_txt)


if __name__ == '__main__':
    convert("/Users/james/IdeaProjects/object_detection/data/val_labels.csv",
            "/Users/james/IdeaProjects/object_detection/data/val_labels.json")
