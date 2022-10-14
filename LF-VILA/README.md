# LF-VILA

By [Yuchong Sun](https://github.com/ycsun1972), [Hongwei Xue](https://hellwayxue.github.io), [Ruihua Song](https://gsai.ruc.edu.cn/addons/teacher/index/info.html?user_id=0&ruccode=20200031&ln=en), [Bei Liu](https://www.microsoft.com/en-us/research/people/libei/), [Huan Yang](https://www.microsoft.com/en-us/research/people/huayan/), [Jianlong Fu](https://www.microsoft.com/en-us/research/people/jianf/).


The repo is the official implemenation of [Long-Form Video-Language Pre-Training with Multimodal Temporal Contrastive Learning](https://arxiv.org/abs/2210.06031).

## Introdution
We introduce a **L**ong-**F**orm **VI**deo-**LA**nguage pre-training model (LF-VILA) and train it on a large-scale long-form video and paragraph dataset constructed from [HD-VILA-100M](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m). 

We propose a **Multimodal Temporal Contrastive (MTC)** loss to learn the temporal relation across different modalities by encouraging fine-grained alignment between long-form videos and paragraphs. We then propose a **Hierarchical Temporal Window Attention (HTWA)** mechanism to effectively capture long-range dependency while reducing computational cost in Transformer.

We fine-tune the pre-trained LF-VILA model on seven downstream long-form video-language understanding tasks of paragraph-to-video retrieval and long-form video question-answering, and achieve new state-of-the-art performances. Specifically, our model achieves **16.1\%**  relative improvement on ActivityNet paragraph-to-video retrieval task and **2.4\%** on How2QA task, respectively.


<p align="center">
<img src="figs/framework.png" alt="framework" width="80%"/>
</p>
<p align="center">
<font size=2 color="gray">The framework of LF-VILA.</font>
</p>

We will release our code, dataset, and pre-trained models soon. Thanks for your patience.

## Citing LF-VILA

If you find the code and pre-trained models useful for your research, please consider citing our paper. :blush:

```bibtex
@inproceedings{sun2022long-form,
    title={Long-Form Video-Language Pre-Training with Multimodal Temporal Contrastive Learning},
    author={Sun, Yuchong and Xue, Hongwei and Song, Ruihua and Liu, Bei and Yang, Huan and Fu, Jianlong},
    booktitle={NeurIPS},
    year={2022}
}
```

## Acknowledgements
The code is based on [HD-VILA](https://github.com/microsoft/XPretrain/tree/main/hd-vila), [ClipBERT](https://github.com/jayleicn/ClipBERT), [FROZEN](https://github.com/m-bain/frozen-in-time), and [Huggingface Transformers](https://github.com/huggingface/transformers), thanks to them! 
