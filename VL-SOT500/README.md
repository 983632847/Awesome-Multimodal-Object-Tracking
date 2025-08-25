# COST: Contrastive One-Stage Transformer for Vision-Language Small Object Tracking [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253525006761)][[arXiv](https://arxiv.org/abs/2504.01321)]

Transformer has recently demonstrated great potential in improving vision-language (VL) tracking algorithms. However, most of the existing VL trackers rely on carefully designed mechanisms to perform the multi-stage multi-modal fusion. Additionally, direct multi-modal fusion without alignment ignores distribution discrepancy between modalities in feature space, potentially leading to suboptimal representations. In this work, we propose COST, a contrastive one-stage transformer fusion framework for VL tracking, aiming to learn semantically consistent and unified VL representations. Specifically, we introduce a contrastive alignment strategy that maximizes mutual information (MI) between a video and its corresponding language description. This enables effective cross-modal alignment, yielding semantically consistent features in the representation space. By leveraging a visual-linguistic transformer, we establish an efficient multi-modal fusion and reasoning mechanism, empirically demonstrating that a simple stack of transformer encoders effectively enables unified VL representations. Moreover, we contribute a newly collected VL tracking benchmark dataset for small object tracking, named VL-SOT500, with bounding boxes and language descriptions. Our dataset comprises two challenging subsets, VL-SOT230 and VL-SOT270, dedicated to evaluating generic and high-speed small object tracking, respectively. Small object tracking is notoriously challenging due to weak appearance and limited features, and this dataset is, to the best of our knowledge, the first to explore the usage of language cues to enhance visual representation for small object tracking. Extensive experiments demonstrate that COST achieves state-of-the-art performance on five existing VL tracking datasets, as well as on our proposed VL-SOT500 dataset.

## VL-SOT500
<div align="center">
<img src="https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/VL-SOT500/figs/Examples_of_VL_SOT500.png" width="600">
<p align="center">
</p>
</div>

<div align="center">
<img src="https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/VL-SOT500/figs/Statistics_Analysis.png" width="600">
<p align="center">
</p>
</div>

### Download Dataset
- Download the VL-SOT500 dataset through [Baidu Pan](https://pan.baidu.com/s/1Q33zh8rWErODN9gJHNP48A?pwd=S500), the extraction code is ***S500***.

The directory should have the following format:
```
├── VL-SOT230
    ├── Video-1
        ├── 000001.jpg
        ├── imgs
            ├── 000001.jpg
            ├── 000002.jpg
            ├── 000003.jpg
            ...
        ├── absent.txt
        ├── attributes.txt
        ├── groundtruth_rect.txt
        ├── language.txt

    ├── Video-2
    ├── Video-3
    ...

├── VL-SOT270
    ├── Video-1
        ├── 000001.jpg
        ├── imgs
            ├── 000001.jpg
            ├── 000002.jpg
            ├── 000003.jpg
            ...
        ├── absent.txt
        ├── attributes.txt
        ├── groundtruth_rect.txt
        ├── language.txt
        ├── relative_speed.txt

    ├── Video-2
    ├── Video-3
    ...

```



## COST
<div align="center">
<img src="https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/VL-SOT500/figs/COST.png" width="600">
<p align="center">
</p>
</div>

The source code will be released soon.


## Evaluation Toolkit 
* Download baseline results through [Baidu Pan](https://pan.baidu.com/s/1rVbQKjMdz-YnQx03LFBuHw?pwd=Base). 
* The [MMOT Evaluation Toolkit](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/MMOT_Evaluation_Toolkit) now supports evaluation methods on the VL-SOT500 dataset.

<div align="center">
<img src="https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/VL-SOT500/figs/VL-SOT270-Result1.png" width="600">
<p align="center">
</p>
</div>

<div align="center">
<img src="https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/VL-SOT500/figs/VL-SOT270-Result2.png" width="600">
<p align="center">
</p>
</div>

### :newspaper: Citation 
If you think this paper is helpful, please feel free to leave a star ⭐ and cite our paper:
```bibtex
@article{zhang2025cost,
  title={COST: Contrastive One-Stage Transformer for Vision-Language Small Object Tracking},
  author={Zhang, Chunhui and Liu, Li and Gao, Jialin and Sun, Xin and Wen, Hao and Zhou, Xi and Ge, Shiming and Wang, Yanfeng},
  journal={arXiv preprint arXiv:2504.01321},
  year={2025}
}
```
