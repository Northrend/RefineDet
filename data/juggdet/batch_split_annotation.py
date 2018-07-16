import os
import subprocess
import sys

HOMEDIR = os.path.expanduser("~")
CURDIR = os.path.dirname(os.path.realpath(__file__))

### Modify the address and parameters accordingly ###
# If true, redo the whole thing.
redo = True
# The root directory which stores the coco images, annotations, etc.
coco_data_dir = '/workspace/dataset/blademaster/juggernaut/juggdet/juggdet_0503' 
# The sets that we want to split. These can be downloaded at: http://mscoco.org
# Unzip all the files after download.
anno_sets = ['juggdet_0503_train_0712']
# These are the sets that used in ION by Sean Bell and Ross Girshick.
# These can be downloaded at: https://github.com/rbgirshick/py-faster-rcnn/tree/master/data
# Unzip all the files after download. And move them to annotations/ directory.
# anno_sets = anno_sets + ["instances_minival2014", "instances_valminusminival2014"]
# The directory which contains the full annotation files for each set.
anno_dir = "{}/Lists/annotations".format(coco_data_dir)
# The root directory which stores the annotation for each image for each set.
out_anno_dir = "{}/Lists/annotations_refinedet_style".format(coco_data_dir)
# The directory which stores the imageset information for each set.
imgset_dir = "{}/Lists".format(coco_data_dir)

### Process each set ###
for i in xrange(0, len(anno_sets)):
    anno_set = anno_sets[i]
    anno_file = "{}/{}.json".format(anno_dir, anno_set)
    if not os.path.exists(anno_file):
        print "{} does not exist".format(anno_file)
        continue
    anno_name = 'juggdet_0503_train_0712' 
    out_dir = "{}/{}".format(out_anno_dir, anno_name)
    imgset_file = "{}/{}.lst".format(imgset_dir, anno_name)
    if redo or not os.path.exists(out_dir):
        cmd = "python {}/split_annotation.py --out-dir={} --imgset-file={} {}" \
                .format(CURDIR, out_dir, imgset_file, anno_file)
        print cmd
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output = process.communicate()[0]
        print output
