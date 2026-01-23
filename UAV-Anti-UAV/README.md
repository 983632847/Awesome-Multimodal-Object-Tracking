# How Far are Modern Trackers from UAV-Anti-UAV? A Million-Scale Benchmark and New Baseline [[Paper](https://arxiv.org/abs/2512.07385)] [[中文解读](https://mp.weixin.qq.com/s/vRVjzzeB_8gyhEmkD6jRlQ)]
### Abstract

Unmanned Aerial Vehicles (UAVs) offer wide-ranging applications but also pose significant safety and privacy violation risks in areas like airport and infrastructure inspection, spurring the rapid development of Anti-UAV technologies in recent years. However, current Anti-UAV research primarily focuses on RGB, infrared (IR), or RGB-IR videos captured by fixed ground cameras, with little attention to tracking target UAVs from another moving UAV platform. To fill this gap, we propose a new multi-modal visual tracking task termed UAV-Anti-UAV, which involves a pursuer UAV tracking a target adversarial UAV in the video stream. Compared to existing Anti-UAV tasks, UAV-Anti-UAV is more challenging due to severe dual-dynamic disturbances caused by the rapid motion of both the capturing platform and the target. To advance research in this domain, we construct a million-scale dataset consisting of 1,820 videos, each manually annotated with bounding boxes, a language prompt, and 15 tracking attributes. Furthermore, we propose MambaSTS, a Mamba-based baseline method for UAV-Anti-UAV tracking, which enables integrated spatial-temporal-semantic learning. Specifically, we employ Mamba and Transformer models to learn global semantic and spatial features, respectively, and leverage the state space model's strength in long-sequence modeling to establish video-level long-term context via a temporal token propagation mechanism. We conduct experiments on the UAV-Anti-UAV dataset to validate the effectiveness of our method. A thorough experimental evaluation of 50 modern deep tracking algorithms demonstrates that there is still significant room for improvement in the UAV-Anti-UAV domain. The dataset and codes will be available at here.

### TODO
- 🚧 Codes for MambaSTS (We are actively improving and expanding MambaSTS for a potential submission)
- ✅ [UAV-Anti-UAV dataset V1.5](https://pan.baidu.com/s/139xn-nKY4KbTOupCn2XDyg?pwd=UAVU) (Contains 1,820 videos in total, with 1,400 allocated for training and 420 for testing)
- ✅ [Baseline Results](https://pan.baidu.com/s/139xn-nKY4KbTOupCn2XDyg?pwd=UAVU)
- ✅ [Evaluation Toolkits](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/MMOT_Evaluation_Toolkit)
- ✅ [Technical Report V1.0](https://arxiv.org/abs/2512.07385)


## UAV-Anti-UAV Dataset

![image](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/UAV-Anti-UAV/UAV-Anti-UAV.png)

#### Step 1: Download dataset
- Download the UAV-Anti-UAV through [Baidu Pan](https://pan.baidu.com/s/139xn-nKY4KbTOupCn2XDyg?pwd=UAVU), the extraction code is ***UAVU***.
#### Step 2: Extract frames from videos: run python [Videos2Frames.py](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/WebUOT-1M/Videos2Frames.py)

The directory should have the following format:
```
├── UAV-Anti-UAV
    ├──Test
        ├── UAV-Anti-UAV_Test_000001
            ├── UAV-Anti-UAV_Test_000001.mp4
            ├── imgs
                ├── 00000001.jpg
                ├── 00000002.jpg
                ├── 00000003.jpg
                ...
            ├── groundtruth_rect.txt
            ├── language.txt
            ├── attributes.txt
            ├── absent.txt
            ├── UAV-Anti-UAV_Test_000001.jpg
        ├── UAV-Anti-UAV_Test_000002
        ├── UAV-Anti-UAV_Test_000003
        ...

    ├──Train
        ├── UAV-Anti-UAV_Train_000001
        ├── UAV-Anti-UAV_Train_000002
        ...
```


## MambaSTS
![image](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/UAV-Anti-UAV/MambaSTS.png)



### BibTeX
If you find our dataset and method both interesting and helpful, please consider citing us in your research or publications:

    @article{zhang2025far,
      title={How Far are Modern Trackers from UAV-Anti-UAV? A Million-Scale Benchmark and New Baseline},
      author={Zhang, Chunhui and Liu, Li and Zhang, Zhipeng and Wang, Yong and Wen, Hao and Zhou, Xi and Ge, Shiming and Wang, Yanfeng},
      journal={arXiv preprint arXiv:2512.07385},
      year={2025}
    }

