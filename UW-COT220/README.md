# Towards Underwater Camouflaged Object Tracking: Benchmark and Baselines [[Paper](https://arxiv.org/abs/2409.16902)] [[ResearchGate](https://www.researchgate.net/publication/388189638_Towards_Underwater_Camouflaged_Object_Tracking_Benchmark_and_Baselines)]

### Abstract

Over the past decade, significant progress has been made in visual object tracking, largely due to the availability of large-scale datasets. However, existing tracking datasets are primarily focused on open-air scenarios, which greatly limits the development of object tracking in underwater environments. To bridge this gap, we take a step forward by proposing the first large-scale multimodal underwater camouflaged object tracking dataset, namely UW-COT220. Based on the proposed dataset, this paper first comprehensively evaluates current advanced visual object tracking methods and SAM- and SAM2-based trackers in challenging underwater environments. Our findings highlight the improvements of SAM2 over SAM, demonstrating its enhanced ability to handle the complexities of underwater camouflaged objects. Furthermore, we propose a novel vision-language tracking framework called VL-SAM2, based on the video foundation model SAM2. Experimental results demonstrate that our VL-SAM2 achieves state-of-the-art performance on the UW-COT220 dataset. The dataset and codes can be accessible at here.

### TODO
- [x] UW-COT220 (The First Multimodal UnderWater Camouﬂaged Object Tracking Dataset)
- [x] Baseline Results
- [x] Evaluation Toolkits
- [ ] Codes for VL-SAM2 (A new VL tracker)

## UW-COT220

![image](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/UW-COT220/UW-COT220.png)

#### Step 1: Download dataset
- Download the UW-COT220 through [Baidu Pan](https://pan.baidu.com/s/1kQH09jmRpieuZsfNeAayjw?pwd=UCOT), the extraction code is ***UCOT***.
- Or download the UW-COT220 through [Google Drive](https://drive.google.com/drive/folders/1iQFdRnmQOUH6tey-RuW63Ck8Nb0RWN-d?usp=sharing).
#### Step 2: Extract frames from videos: run python [Videos2Frames.py](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/WebUOT-1M/Videos2Frames.py)

The directory should have the below format:
```
├── UW-COT220

    ├── UWCOT220_000001
        ├── UWCOT220_000001.mp4
            ├── imgs
            ├── 00000001.jpg
            ├── groundtruth_rect.txt
            ├── language.txt
            ├── masks (Coming Soon)
    ├── UWCOT220_000002
    ├── UWCOT220_000003
      ...
```

## VL-SAM2
![image](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/UW-COT220/VL-SAM2.png)


## MMOT Evaluation Toolkit
The tutorial of [MMOT Evaluation toolkit](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/MMOT_Evaluation_Toolkit).

The baseline results can be downloaded from our evaluation toolkit.


## Thanks
This implementation is based on [SAM2](https://github.com/facebookresearch/segment-anything-2), [OKTrack](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/WebUOT-1M), and [SAMURAI](https://github.com/yangchris11/samurai). Please refer to their repositories for more details.


### BibTeX
If you find our dataset and method both interesting and helpful, please consider citing us in your research or publications:

    @article{zhang2024towards,
        title={Towards Underwater Camouflaged Object Tracking: Benchmark and Baselines},
        author={Zhang, Chunhui and Liu, Li and Huang, Guanjie and Wen, Hao and Zhou, Xi and Wang, Yanfeng},
        journal={arXiv preprint arXiv:2409.16902},
        year={2024}
    }

