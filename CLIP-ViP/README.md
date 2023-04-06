# CLIP-ViP (ICLR 2023)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-vip-adapting-pre-trained-image-text/video-retrieval-on-activitynet)](https://paperswithcode.com/sota/video-retrieval-on-activitynet?p=clip-vip-adapting-pre-trained-image-text)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-vip-adapting-pre-trained-image-text/video-retrieval-on-didemo)](https://paperswithcode.com/sota/video-retrieval-on-didemo?p=clip-vip-adapting-pre-trained-image-text)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-vip-adapting-pre-trained-image-text/video-retrieval-on-lsmdc)](https://paperswithcode.com/sota/video-retrieval-on-lsmdc?p=clip-vip-adapting-pre-trained-image-text)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-vip-adapting-pre-trained-image-text/video-retrieval-on-msr-vtt-1ka)](https://paperswithcode.com/sota/video-retrieval-on-msr-vtt-1ka?p=clip-vip-adapting-pre-trained-image-text)


By [Hongwei Xue](https://hellwayxue.github.io/)\*, [Yuchong Sun](https://scholar.google.com/citations?user=DuSxNqgAAAAJ&hl=en)\*, [Bei Liu](https://www.microsoft.com/en-us/research/people/libei/), [Jianlong Fu](https://www.microsoft.com/en-us/research/people/jianf/), [Ruihua Song](https://scholar.google.com/citations?user=v5LctN8AAAAJ&hl=en), [Houqiang Li](http://staff.ustc.edu.cn/~lihq/en/), [Jiebo Luo](https://www.cs.rochester.edu/u/jluo/).


This repo is the official pytorch implementation of [CLIP-ViP: Adapting Image-Text Pre-training to Video-Language Representation Learning](https://arxiv.org/abs/2209.06430), accepted by [ICLR 2023](https://iclr.cc/Conferences/2023). CLIP-ViP is a video-language model which is based on a pre-trained image-text model [CLIP](https://openai.com/blog/clip/) then further pre-trained (post-pretraining) on a large-scale video-text dataset [HD-VILA-100M](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m). This repo consists of the code of CLIP-ViP model, the post-pretraining method, finetuning on text-to-video retrieval.


## Requirements 
We provide a Docker image for easier reproduction: `tiankaihang/azureml_docker:horovod`

We use mixed-precision training hence GPUs with Tensor Cores are recommended.


## Getting Started

### General

1. Download Data.

    Download HD-VILA-100M and other required data following the instruction of [HD-VILA](https://github.com/microsoft/XPretrain/tree/main/hd-vila). Also download auxiliary captions from: [Azure Blob Link](https://hdvila.blob.core.windows.net/dataset/hdvila_ofa_captions_db.zip?sp=r&st=2023-03-16T04:58:26Z&se=2026-03-01T12:58:26Z&spr=https&sv=2021-12-02&sr=b&sig=EYE%2Bj11VWfQ6G5dZ8CKlOOpL3ckmmNqpAtUgBy3OGDM%3D)

2. Download pretrained models.

    We release the CLIP-ViP model under two settings:

    CLIP-ViP-B/32: [Azure Blob Link](https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_32.pt?sp=r&st=2023-03-16T05:02:41Z&se=2027-05-31T13:02:41Z&spr=https&sv=2021-12-02&sr=b&sig=91OEG2MuszQmr16N%2Bt%2FLnvlwY3sc9CNhbyxYT9rupw0%3D)

    CLIP-ViP-B/16: [Azure Blob Link](https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_16.pt?sp=r&st=2023-03-16T05:02:05Z&se=2026-07-31T13:02:05Z&spr=https&sv=2021-12-02&sr=b&sig=XNd7fZSsUhW7eesL3hTfYUMiAvCCN3Bys2TadXlWzFU%3D)

3. Set up the environment for running the experiments.

    Create a folder that stores pretrained models, all the data, and results.
    ```bash
    PATH_TO_STORAGE=/path/to/your/data/
    ```
    Clone this repo and launch the Docker container for running the experiments. 
    If you want to pre-train on your own dataset, please prepare the environment with `horovod`. It is a better choice to use the pre-built docker image `tiankaihang/azureml_docker:horovod`. Or you can build from the [dockerfile](./docker/Dockerfile).
    We use mixed-precision training hence GPUs with Tensor Cores are recommended.
    ```bash
    # command to get into the container, docker image should be automatically pulled.
    cd HD-VILA
    source launch_container.sh $PATH_TO_STORAGE
    # update the transformers package
    pip install --upgrade transformers
    ```


### Pre-training

```bash
#inside the container
horovodrun -np $NUM_GPUS python src/pretrain/run_pretrain.py \
    --config $CONFIG_PATH
``` 

`$CONFIG_PATH` should be set to one of the .json config files available at [src/configs/pretrain](src/configs/pretrain). Currently, `pretrain_vip_base_32.json` and `pretrain_vip_base_16.json` are supported

### Text-to-Video Retrieval Finetuning

```bash
# inside the container
horovodrun -np $NUM_GPUS python src/tasks/run_video_retrieval.py \
    --config $CONFIG_PATH 
```

`$CONFIG_PATH` should be set to one of the .json config files available at [src/configs](src/configs) postfixed with `_retrieval.json`. For example, you can use `src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json` for finetuning CLIP-ViP-B/32 on MSRVTT retrieval. For model, currently, `pretrain_vip_base_32.json` and `pretrain_vip_base_16.json` are supported. For dataset, MSR-VTT, DiDemo, LSMDC, ActivityNet Captions are supported.


## Citation
If you find the code and pre-trained models useful for your research, please consider citing our paper:

```
@article{xue2022clip,
  title={CLIP-ViP: Adapting Pre-trained Image-Text Model to Video-Language Representation Alignment},
  author={Xue, Hongwei and Sun, Yuchong and Liu, Bei and Fu, Jianlong and Song, Ruihua and Li, Houqiang and Luo, Jiebo},
  journal={arXiv preprint arXiv:2209.06430},
  year={2022}
}

@inproceedings{xue2022advancing,
  title={Advancing high-resolution video-language representation with large-scale video transcriptions},
  author={Xue, Hongwei and Hang, Tiankai and Zeng, Yanhong and Sun, Yuchong and Liu, Bei and Yang, Huan and Fu, Jianlong and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5036--5045},
  year={2022}
}
```

## Acknowledgements
The code is based on [HD-VILA](https://github.com/microsoft/XPretrain/tree/main/hd-vila).
