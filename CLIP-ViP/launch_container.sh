DATA_DIR=$1

if [ -z $CUDA_VISIBLE_DEVICES ]; then
   CUDA_VISIBLE_DEVICES='all'
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
   --mount src=$(pwd),dst=/VidCLIP,type=bind \
   --mount src=$DATA_DIR,dst=/blob_mount,type=bind \
   -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
   -w /VidCLIP tiankaihang/azureml_docker:horovod \
   bash -c "source /VidCLIP/setup.sh && export OMPI_MCA_btl_vader_single_copy_mechanism=none && bash"

