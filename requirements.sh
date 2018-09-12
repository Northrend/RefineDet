apt-get update
apt-get install --no-install-recommends libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libboost-all-dev libgflags-dev libgoogle-glog-dev liblmdb-dev libatlas-base-dev

pip install --upgrade pip
for req in $(cat ./python/requirements.txt); do pip install $req -i https://pypi.tuna.tsinghua.edu.cn/simple; done
