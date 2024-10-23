# WebUOT-1M: Advancing Deep Underwater Object Tracking with A Million-Scale Benchmark [[Paper](https://arxiv.org/abs/2405.19818)]

### Abstract

Underwater object tracking (UOT) is a foundational task for identifying and tracing submerged entities in underwater video sequences. However, current UOT datasets suffer from limitations in scale, diversity of target categories and scenarios covered, hindering the training and evaluation of modern tracking algorithms. To bridge this gap, we take the first step and introduce WebUOT-1M, \ie, the largest public UOT benchmark to date, sourced from complex and realistic underwater environments. It comprises 1.1 million frames across 1,500 video clips filtered from 408 target categories, largely surpassing previous UOT datasets, \eg, UVOT400. Through meticulous manual annotation and verification, we provide high-quality bounding boxes for underwater targets. Additionally, WebUOT-1M includes language prompts for video sequences, expanding its application areas, \eg, underwater vision-language tracking. Most existing trackers are tailored for open-air environments, leading to performance degradation when applied to UOT due to domain gaps. Retraining and fine-tuning these trackers are challenging due to sample imbalances and limited real-world underwater datasets. To tackle these challenges, we propose a novel omni-knowledge distillation framework based on WebUOT-1M, incorporating various strategies to guide the learning of the student Transformer. To the best of our knowledge, this framework is the first to effectively transfer open-air domain knowledge to the UOT model through knowledge distillation, as demonstrated by results on both existing UOT datasets and the newly proposed WebUOT-1M. Furthermore, we comprehensively evaluate WebUOT-1M using 30 deep trackers, showcasing its value as a benchmark for UOT research by presenting new challenges and opportunities for future studies. The complete dataset, codes and tracking results, will be made publicly available.

### TODO
- [ ] WebUOT-1M Dataset
- [x] Baseline Results
- [x] Evaluation Toolkits
- [x] Codes and Weights for OKTrack
- [ ] Codes and Weights for OKTrack++ (VL version)


## WebUOT-1M

![image](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/WebUOT-1M/WebUOT-1M.png)


## OKTrack
- Download the OKTrack through [Baidu Pan](https://pan.baidu.com/s/1j3i_znyWOo9MI7I6_1tltA?pwd=OKTK), the extraction code is ***OKTK***.
- Download the OKTrack through [Google Drive](todo).
![image](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/WebUOT-1M/OKTrack.png)


#### Evaluation   
```
python tracking/test.py
python tracking/analysis_results.py
```
Before evaluation, please make sure the data path in [***local.py***](./lib/test/evaluation/local.py) is correct.
We recommend using the [MMOT Evaluation toolkit](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/MMOT_Evaluation_Toolkit).


#### Training

1.Training with one GPU.
```
cd /$PROJECT_ROOT$/OKTrack/lib/train
python run_training.py --save_dir ./output
```

2.Training with multiple GPUs.
```
cd /$PROJECT_ROOT$/OKTrack
python tracking/train.py --save_dir ./output --mode multiple --nproc_per_node 8
```

Before training, please make sure the data path in [***local.py***](./lib/train/admin/local.py) is correct.


## MMOT Evaluation Toolkit
The tutorial of [MMOT Evaluation toolkit](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/MMOT_Evaluation_Toolkit).

The baseline results can be downloaded from our evaluation toolkit.


## Thanks
This implementation is based on [OSTrack](https://github.com/botaoye/OSTrack), [HDETrack](https://github.com/Event-AHU/EventVOT_Benchmark), and [All-in-One](https://github.com/983632847/All-in-One). Please ref to their reposity for more details.


Our dataset and codes are currently in preparation and will be released shortly!


### BibTeX
If you find our dataset and method both interesting and helpful, please consider citing us in your research or publications:

    @article{zhang2024webuot,
      title={WebUOT-1M: Advancing Deep Underwater Object Tracking with A Million-Scale Benchmark},
      author={Zhang, Chunhui and Liu, Li and Huang, Guanjie and Wen, Hao and Zhou, Xi and Wang, Yanfeng},
      journal={arXiv preprint arXiv:2405.19818},
      year={2024}
    }
