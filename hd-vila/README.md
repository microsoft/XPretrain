# HD-VILA

By [Hongwei Xue](https://hellwayxue.github.io/)\*, [Tiankai Hang](https://tiankaihang.github.io/)\*, [Yanhong Zeng](https://1900zyh.github.io/)\*, [Yuchong Sun](https://github.com/ycsun1972)\*, [Bei Liu](https://www.microsoft.com/en-us/research/people/libei/), [Huan Yang](https://www.microsoft.com/en-us/research/people/huayan/), [Jianlong Fu](https://www.microsoft.com/en-us/research/people/jianf/), and [Baining Guo](https://www.microsoft.com/en-us/research/people/bainguo/).


The repo is the official implemenation of ["Advancing High-Resolution Video-Language Representation with Large-Scale Video Transcriptions"](https://arxiv.org/abs/2111.10337).
It currently contains code and models for 

> Large-scale pre-training on [HD-VILA-100M](../hd-vila-100m/README.md)

> Downstream tasks including video-text retrieval, video question answering.


## Introdution
We propose a novel **H**igh-resolution
and **D**iversified **VI**deo-**LA**nguage pre-training model (HD-VILA) for many visual tasks. 
To enable VL pre-training, we jointly
optimize the HD-VILA model by a hybrid Transformer
that learns rich spatiotemporal features, and a multimodal
Transformer that enforces interactions of the learned video
features with diversified texts. Our pre-training model
achieves new state-of-the-art results in **10** VL understanding tasks and **2** more novel text-to-visual generation tasks.
We outperform SOTA models with relative increases of 40.4% R@1 in zero-shot MSR-VTT text-to-video retrieval task, and 55.4% in high-resolution dataset LSMDC. The learned VL embedding is also effective in generating visually pleasing and semantically relevant results in text-to-visual editing and super-resolution tasks.

<p align="center">
<img src="figs/framework.png" alt="statistics" width="60%"/>
</p>
<p align="center">
<font size=2 color="gray">The framework of HD-VILA.</font>
</p>

## Getting Started


### General

1. Create a folder that stores pretrained models, all the data, and results.
    ```bash
    PATH_TO_STORAGE=/path/to/your/data/
    ```

2. Clone this repo and launch the Docker container for running the experiments. 
    If you want to pre-train on your own dataset, please prepare the environment with `horovod`. It is a better choice to use the pre-built docker image `tiankaihang/azureml_docker:horovod`. Or you can build from the [dockerfile](./docker/Dockerfile).
    We use mixed-precision training hence GPUs with Tensor Cores are recommended.
    ```bash
    # command to get into the container, docker image should be automatically pulled.
    cd HD-VILA
    source launch_container.sh $PATH_TO_STORAGE
    ```

### Prepare data and pretrained model

1. Use [scripts/download_data.sh](scripts/download_data.sh) to download our pretrained model. In stage-one pretraining, we load pre-trained ResNet50 and BERT-large. Our scripts will also download these model weights. 

```bash
    bash scripts/download_data.sh $PATH_TO_STORAGE
```

2. The above script will also download the annotations for all downstream tasks.

+ For Pre-training dataset, please refer to [HD-VILA-100M](../hd-vila-100m/README.md) to prepare pre-training data. Please download the annotations and videos, then put them to $PATH_TO_STORAGE/data/hdvila.

+ For MSRVTT, the official data and video links can be found in [link](http://ms-multimedia-challenge.com/2017/dataset). For convience, you can download the raw videos from the sharing by [Frozen️ in Time](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip). Please download the raw videos and put them to $PATH_TO_STORAGE/data/msrvtt_retrieval/videos. We compresse the frame rate to speed up data reading, the compressed videos will be stored in $PATH_TO_STORAGE/data/msrvtt_retrieval/videos_6fps.

+ For ActivityNet, you can download the raw videos refer to the [official website](http://activity-net.org/download.html). We use the split as [collaborative-experts](https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/activity-net). Please download the raw videos and put them to $PATH_TO_STORAGE/data/activitynet_retrieval/videos. We decode the frame from the original video to speed up data reading. The decoded frames will be stored in $PATH_TO_STORAGE/data/activitynet_retrieval/video_frames and $PATH_TO_STORAGE/data/activitynet_retrieval/video_frames_lr.

+ For DiDeMo, the raw videos can be download from [LisaAnne/LocalizingMoments](https://github.com/LisaAnne/LocalizingMoments). We use the split as [collaborative-experts](https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/activity-net). We decode the frame from the original video to speed up data reading the same as ActivityNet.

+ For LSMDC, the raw videos can be download from the [official website](https://sites.google.com/site/describingmovies/download). We use the official split. We decode the frame from the original video to speed up data reading the same as ActivityNet.

+ For TGIF-QA, you can download the raw videos from the [official page](https://github.com/YunseokJANG/tgif-qa/blob/master/dataset/README.md). Please download the raw videos and put them to $PATH_TO_STORAGE/data/tgif_qa, and then use [gifmp4.py] to convert the videos to MP4(scripts/process_raw_video/gifmp4.py).

+ For MSRVTT-QA and MSRVTT-MC, we use the videos the same as MSRVTT Retrieval task.

+ To speed up data reading, we compresse the frame rate or decode the frame from the original video. We provide the processing code [here](scripts/process_raw_video).

### Pre-training

Our pre-training consists of two stages. Stage one use contrastive learning to train the dual encoder. Then stage two use the masked modeling to train the cross-modal Transformer. During stage two, the parameters in stage one are frozen.

For stage one:
```bash
#inside the container
horovodrun -np $NUM_GPUS python src/pretrain/run_pretrain_stage1_group.py \
    --config src/configs/pretrain_stage1.json 
``` 

For stage two:
```bash
#inside the container
horovodrun -np $NUM_GPUS python src/pretrain/run_pretrain_stage2_group.py \
    --config src/configs/pretrain_stage2.json 
``` 
Notice that replace the `e2e_weights_path` in pretrain_stage2 with your stage one path.

### Finetune the model for downstream tasks
Tasks: MSRVTT, DiDeMo, LSMDC and ActivityNet video-text retrieval, MSRVTT MC Test. MSRVTT and T-GIF QA.
Please replace the data path in [src/configs](src/configs) with your own data path.

1. Finetuning Retrieval.
    ```bash
    # inside the container
    horovodrun -np $NUM_GPUS python src/tasks/run_video_retrieval.py \
        --config $CONFIG_PATH 
    ```
    `$CONFIG_PATH` should be set to one of the .json config files available at [src/configs](src/configs) postfixed with `_retrieval.json`. For example, you can use `src/configs/msrvtt_retrieval.json` for MSRVTT retrieval.

2. Finetuning Video QA.
    ```bash
    # inside the container
    horovodrun -np $NUM_GPUS python src/tasks/run_video_qa.py \
        --config $CONFIG_PATH 
    ```
    `$CONFIG_PATH` should be set to one of the .json config files available at [src/configs](src/configs) postfixed with `_qa.json`. For example, you can use `src/configs/msrvtt_qa.json` 
    for MSRVTT Video QA.

3. MSRVTT Multi-choice Inference.
    After MSRVTT retrieval model is trained, you can use it for inference 
    for the MSRVTT MC Test task as well, which is essentially a retrieval 
    task in a multiple-choice task setup. 
    ```bash
    # inside the container
    horovodrun -np $NUM_GPUS python src/tasks/run_msrvtt_mc.py \
      --config src/configs/msrvtt_retrieval.json
      --do_inference 1 --output_dir $OUTPUT_DIR \
      --inference_split val --inference_model_step $STEP \
      --inference_txt_db /path/to/msrvtt_retrieval/mc_test.jsonl \
      --inference_img_db /path/to/msrvtt_video --inference_batch_size 64 \
      --inference_n_clips $INFERENCE_N_CLIPS
    ```


## License

The license of the code and pre-trained models is [here](./LICENSE).

## Citing HD-VILA

If you find the code and pre-trained models useful for your research, please consider citing our paper. :blush:

```bibtex
@inproceedings{xue2022advancing,
    title={Advancing High-Resolution Video-Language Representation with Large-Scale Video Transcriptions},
    author={Xue, Hongwei and Hang, Tiankai and Zeng, Yanhong and Sun, Yuchong and Liu, Bei and Yang, Huan and Fu, Jianlong and Guo, Baining},
    booktitle={International Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```

## Acknowledgements
The code is based on [ClipBERT](https://github.com/jayleicn/ClipBERT), thanks to them! 
