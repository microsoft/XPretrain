# HD-VILA-100M Dataset

## What is HD-VILA-100M?
HD-VILA-100M is a large-scale, high-resolution, and
diversified video-language dataset to facilitate the multimodal representation learning. 

<p align="center">
<img src="figs/examples.png" alt="examples for hd-vila"/>
</p>
<p align="center">
<font size=2 color="gray">Examples of video clips and ASR generated transcriptions in the proposed HD-VILA-100M dataset.</font>
</p>

## Data statistics
The dataset contains 3.3 million videos in total, which are of high quality and distributed in 15 categories in balance.
<p align="center">
<img src="figs/statics.png" alt="statistics" width="60%"/>
</p>
<p align="center">
<font size=2 color="gray">The distribution of categories in HD-VILA-100M dataset.</font>
</p>

The details of our dataset are presented in the table below.
| Dataset | Domain |  #Video clips | #Sentence | Avg len(sec) | Sent len | Duration(h) | Resolution
| :-----| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| HD-VILA-100M | open | 100M | 100M | 13.4 | 32.5 | 371.5K | 720p |


## Download

You can download all the urls through this [link](https://hdvila.blob.core.windows.net/dataset/hdvila100m.zip?sp=r&st=2022-06-28T03:33:11Z&se=2026-01-01T11:33:11Z&spr=https&sv=2021-06-08&sr=b&sig=VaqQkLFDqKinfkaPNs1jJ1EQIYCB%2FUPYiqFqmjWye6Y%3D) (updated 6/28/2022). Together we also offer all the timestamps to divide the videos into clips. The format of the data is:
```
{   
    'video_id':'QMi8x8o55Ns',
    'url': 'https://www.youtube.com/watch?v=QMi8x8o55Ns',
    'clip': [
                {'clip_id': 'QMi8x8o55Ns.1.mp4', 'span': ['00:00:17.759', '00:00:23.279']}
                ...
                {'clip_id': 'QMi8x8o55Ns.16.mp4', 'span': ['00:04:52.140', '00:05:03.350']}
            ],
}
```

You can download the raw videos from YouTube and use [src/cut_videos.py](./src/cut_videos.py) to cut the videos to clips.


## License

The license of the collected dataset is [here](./LICENSE).

## Citing HD-VILA

If you find this dataset useful for your research, please consider citing our paper. :blush:

```bibtex
@inproceedings{xue2022hdvila,
    title={Advancing High-Resolution Video-Language Representation with Large-Scale Video Transcriptions},
    author={Xue, Hongwei and Hang, Tiankai and Zeng, Yanhong and Sun, Yuchong and Liu, Bei and Yang, Huan and Fu, Jianlong and Guo, Baining},
    booktitle={International Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```

## Contact Information

For further request about dataset or problems using the dataset, you can contact [Bei Liu]() (`bei.liu@microsoft.com`).
