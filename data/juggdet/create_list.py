import argparse
import os
from random import shuffle
import shutil
import subprocess
import sys

HOMEDIR = os.path.expanduser("~")
CURDIR = os.path.dirname(os.path.realpath(__file__))

# If true, re-create all list files.
redo = True
# The root directory which holds all information of the dataset.
# data_dir = "{}/data/Object_Detection/coco".format(HOMEDIR)
data_dir = '/workspace/dataset/blademaster/juggernaut/juggdet'
# The directory name which holds the image sets.
imgset_dir = "juggdet_0503/Lists"
# The direcotry which contains the images.
img_dir = "juggdet_0503/Image"
img_ext = "jpg"
# The directory which contains the annotations.
anno_dir = "juggdet_0503/Lists/annotations_refinedet_style"
anno_ext = "json"

train_list_file = "/workspace/dataset/blademaster/juggernaut/juggdet/juggdet_0503/Lists/annotations_refinedet_style/juggdet_0503_train_0712.caffetxt"
# test_list_file = "{}/juggdet_0503_test.txt".format(CURDIR)

# Create training set.
# We follow Ross Girschick's split.
if redo or not os.path.exists(train_list_file):
    datasets = ['juggdet_0503_train_0712']
    img_files = []
    anno_files = []
    for dataset in datasets:
        imgset_file = "{}/{}/{}.lst".format(data_dir, imgset_dir, dataset)
        with open(imgset_file, "r") as f:
            for line in f.readlines():
                name = os.path.splitext(line.strip("\n"))[0]
                # subset = name.split("_")[1]
                img_file = "{}/{}.{}".format(img_dir, name, img_ext)
                assert os.path.exists("{}/{}".format(data_dir, img_file)), \
                        "{}/{} does not exist".format(data_dir, img_file)
                anno_file = "{}/{}/{}.{}".format(anno_dir, dataset, name, anno_ext)
                assert os.path.exists("{}/{}".format(data_dir, anno_file)), \
                        "{}/{} does not exist".format(data_dir, anno_file)
                img_files.append(img_file)
                anno_files.append(anno_file)
    # Shuffle the images.
    idx = [i for i in xrange(len(img_files))]
    shuffle(idx)
    with open(train_list_file, "w") as f:
        for i in idx:
            f.write("{} {}\n".format(img_files[i], anno_files[i]))

# if redo or not os.path.exists(test_list_file):
#     datasets = ["test2015"]
#     subset = "test2015"
#     img_files = []
#     anno_files = []
#     for dataset in datasets:
#         imgset_file = "{}/{}/{}.txt".format(data_dir, imgset_dir, dataset)
#         with open(imgset_file, "r") as f:
#             for line in f.readlines():
#                 name = line.strip("\n")
#                 img_file = "{}/{}/{}.{}".format(img_dir, subset, name, img_ext)
#                 assert os.path.exists("{}/{}".format(data_dir, img_file)), \
#                         "{}/{} does not exist".format(data_dir, img_file)
#                 anno_file = "{}/{}/{}.{}".format(anno_dir, subset, name, anno_ext)
#                 assert os.path.exists("{}/{}".format(data_dir, anno_file)), \
#                         "{}/{} does not exist".format(data_dir, anno_file)
#                 img_files.append(img_file)
#                 anno_files.append(anno_file)
#     with open(test_list_file, "w") as f:
#         for i in xrange(len(img_files)):
#             f.write("{} {}\n".format(img_files[i], anno_files[i]))
