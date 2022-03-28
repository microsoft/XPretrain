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
The dataset contains 3.1M videos in total with high-quality and distributed in 15 categories in balance.
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

You can download all the urls through this [link](https://hdvila.blob.core.windows.net/dataset/hdvila100m.zip?sv=2020-08-04&ss=bqtf&srt=sco&sp=rwdlacuptfxi&se=2022-03-26T15:09:34Z&sig=hqxktUNEK9RuFtZtZMsJMKopbLKp6WTsITdyoByz5y0%3D&_=1648278578850). We also offer all the timestamps to divide the videos into clips. The format of the data is:
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

We will release all the code to download and process the data soon.


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
