cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=/workspace/RefineDet

cd $root_dir

redo=false
# data_root_dir="$HOME/data/Object_Detection/coco"
data_root_dir="/workspace/dataset/blademaster/juggernaut/juggdet/"
dataset_name="juggdet"
mapfile="$root_dir/data/$dataset_name/labelmap_$dataset_name.prototxt"
listfile="/workspace/dataset/blademaster/juggernaut/juggdet/juggdet_0503/Lists/annotations_refinedet_style/juggdet_0503_train.caffetxt"
anno_type="detection"
label_type="json"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0
outputpath="/workspace/dataset/blademaster/juggernaut/juggdet/juggdet_0503/Cache/juggdet_0503_train_$db"

extra_cmd="--encode-type=jpg --encoded"
if $redo
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in train 
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-type=$label_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $listfile $outputpath examples/$dataset_name 2>&1 | tee $root_dir/data/$dataset_name/$subset.log
done
