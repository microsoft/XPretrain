DATA_DIR=$1
NAME=$2


if [ -z $CUDA_VISIBLE_DEVICES ]; then
   CUDA_VISIBLE_DEVICES='all'
fi

docker run --gpus device=$CUDA_VISIBLE_DEVICES --ipc=host --rm -it \
   --name $NAME \
   --shm-size=128g \
   --mount src=$(pwd),dst=/LF-VILA,type=bind \
   --mount src=$DATA_DIR,dst=/blob_mount,type=bind \
   -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
   -w /LF-VILA ycsun1972/azureml_docker:horovod_deepspeed_v2 \
   bash -c "source /LF-VILA/setup.sh && bash"
