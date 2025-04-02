# Underwater Camouflaged Object Tracking Meets Vision-Language SAM2 [[Paper](https://arxiv.org/abs/2409.16902)] [[ResearchGate](https://www.researchgate.net/publication/390421004_Underwater_Camouflaged_Object_Tracking_Meets_Vision-Language_SAM2)]

### Abstract

Over the past decade, significant progress has been made in visual object tracking, largely due to the availability of large-scale datasets. However, these datasets have primarily focused on open-air scenarios and have largely overlooked underwater animal tracking—especially the complex challenges posed by camouflaged marine animals. To bridge this gap, we take a step forward by proposing the first large-scale multi-modal underwater camouflaged object tracking dataset, namely UW-COT220. Based on the proposed dataset, this work first comprehensively evaluates current advanced visual object tracking methods, including SAM- and SAM2-based trackers, in challenging underwater environments, \eg, coral reefs. Our findings highlight the improvements of SAM2 over SAM, demonstrating its enhanced ability to handle the complexities of underwater camouflaged objects. Furthermore, we propose a novel vision-language tracking framework called VL-SAM2, based on the video foundation model SAM2. Experimental results demonstrate that our VL-SAM2 achieves state-of-the-art performance on the UW-COT220 dataset. The dataset and codes are available at here.

### TODO
- [x] UW-COT220 (The First Multimodal UnderWater Camouﬂaged Object Tracking Dataset)
- [x] Baseline Results
- [x] Evaluation Toolkits
- [x] Codes for VL-SAM2 (A new VL tracker)

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

### Train VL-SAM2
#### Step1: Download Training Dataset [Refer-YouTube-VOS](https://youtube-vos.org/dataset/rvos/)
The directory should have the below format:
```
├── Refer-YouTube-VOS
    ├── train
        ├── Annotations
        ├── JPEGImages
        ├── meta.json (do not use)
        ...
     ├── meta_expressions
            ├── train
                ├── meta_expressions.json
            ...
```

#### Step2: Installation SAM2

[SAM 2](https://github.com/facebookresearch/sam2) needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.5.1` and `torchvision>=0.20.1`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. You can install SAM 2 on a GPU machine using:

```bash
cd your_path_to_VL_SAM2/VL-SAM2/
pip install -e .
```

#### Step3: Get Language Embeddings
```bash
cd your_path_to_VL_SAM2/VL-SAM2/
python Get_language_embeddings.py   
```

#### Step4: Train VL-SAM2
```bash
cd your_path_to_VL_SAM2/VL-SAM2/training/
python train.py 
```

### Test VL-SAM2
Download pre-trained weights through [Baidu Pan](https://pan.baidu.com/s/1IFsxW9U0AuZVhPvMQk9FHA?pwd=VLS2), or [Google Drive](https://drive.google.com/drive/folders/1Ob7tSMikRmz54kZRzn_T8QwQMlvo-ugk?usp=sharing).

Then, you can test VL-SAM2 on UW-COT220 by running:

```bash
cd your_path_to_VL_SAM2/VL-SAM2/
python Test_VL_SAM2_UWCOT220.py
```

#### We release VL-SAM2 with the language branch, and the plug-and-play MATP module can be quickly implemented by referring to [SAMURAI](https://github.com/yangchris11/samurai).


## MMOT Evaluation Toolkit
The tutorial of [MMOT Evaluation toolkit](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/MMOT_Evaluation_Toolkit).

The baseline results can be downloaded from our evaluation toolkit.


## Thanks
This implementation is based on [SAM2](https://github.com/facebookresearch/segment-anything-2), [OKTrack](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/WebUOT-1M), and [SAMURAI](https://github.com/yangchris11/samurai). Please refer to their repositories for more details.


### BibTeX
If you find our dataset and method both interesting and helpful, please consider citing us in your research or publications:

    @article{zhang2024towards,
        title={Underwater Camouflaged Object Tracking Meets Vision-Language SAM2},
        author={Zhang, Chunhui and Liu, Li and Huang, Guanjie and Zhang, Zhipeng and Wen, Hao and Zhou, Xi and Ge, Shiming and Wang, Yanfeng},
        journal={arXiv preprint arXiv:2409.16902},
        year={2024}
    }

