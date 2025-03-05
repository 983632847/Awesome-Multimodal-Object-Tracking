[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com) [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![MMOT](https://img.shields.io/badge/Paper-MMOT-2b9348.svg?logo=arXiv)](https://arxiv.org/abs/2405.14200)
# [Awesome Multi-modal Object Tracking](https://arxiv.org/abs/2405.14200) 

<div align="center">
<img src="https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/MMOT.png" width="600">


------
<p align="center">
</p>
</div>

#### A continuously updated project to track the latest progress in multi-modal object tracking (MMOT).
#### If this repository can bring you some inspiration, we would feel greatly honored.
#### If you like our project, please give us a star ⭐ on this GitHub.
#### If you have any suggestions, please feel free to contact: [andyzhangchunhui@gmail.com](andyzhangchunhui@gmail.com). 
### We welcome researchers to submit pull requests and become contributors to this project.
### This project focuses solely on single-object tracking.
### Awesome Visual Object Tracking (VOT) Project is at [Awesome-VOT](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/Awesome_Visual_Object_Tracking).

<p align="center">
<img src="https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/Main_MMOT_Paradigms.png" width="600px"/>  
</p>

## :collision: Highlights
![Last Updated](https://badgen.net/github/last-commit/983632847/Awesome-Multimodal-Object-Tracking?icon=github&label=last%20updated&color=blue)

- 2025.02.28: Awesome Visual Object Tracking Project Started at [Awesome-VOT](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/Awesome_Visual_Object_Tracking).
- 2025.01.20: The Technical Report for UW-COT220 and VL-SAM2 was Updated [arXiv](https://arxiv.org/abs/2409.16902) [知乎](https://zhuanlan.zhihu.com/p/19568835256).
- 2024.09.26: The WebUOT-1M was Accepted by NeurIPS 2024, and its Extended Version, UW-COT220, was Online.
- 2024.05.30: The Paper of WebUOT-1M was Online [arXiv](https://arxiv.org/abs/2405.19818).
- 2024.05.24: The Report of Awesome MMOT Project was Online [arXiv](https://arxiv.org/abs/2405.14200) [知乎](https://zhuanlan.zhihu.com/p/699538389).
- 2024.05.20: Awesome MMOT Project Started.


## Contents

- [Survey](#survey) 
- [Vision-Language Tracking (RGBL Tracking)](#vision-language-tracking)
  - [Datasets](#datasets)
  - [Papers](#papers)
- [RGBE Tracking](#rgbe-tracking)
  - [Datasets](#datasets)
  - [Papers](#papers)
- [RGBD Tracking](#rgbd-tracking)
  - [Datasets](#datasets)
  - [Papers](#papers)
- [RGBT Tracking](#rgbt-tracking)
  - [Datasets](#datasets)
  - [Papers](#papers)
- [Miscellaneous (RGB+X)](#miscellaneous)
  - [Datasets](#datasets)
  - [Papers](#papers)
- [Others](#others)
- [Awesome Repositories for MMOT](#awesome-repositories-for-mmot)
  

## Citation

If you find our work useful in your research, please consider citing:
```
@article{zhang2024awesome,
  title={Awesome Multi-modal Object Tracking},
  author={Zhang, Chunhui and Liu, Li and Wen, Hao and Zhou, Xi and Wang, Yanfeng},
  journal={arXiv preprint arXiv:2405.14200},
  year={2024}
}
```


## Survey
- **Awesome-MMOT:** Chunhui Zhang, Li Liu, Hao Wen, Xi Zhou, Yanfeng Wang.<br />
  "Awesome Multi-modal Object Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2405.14200)] 
  [[project](https://github.com/983632847/Awesome-Multimodal-Object-Tracking)]

- Pengyu Zhang, Dong Wang, Huchuan Lu.<br />
  "Multi-modal Visual Tracking: Review and Experimental Comparison." ArXiv (2022).
  [[paper](https://arxiv.org/abs/2012.04176)] 

- Zhangyong Tang, Tianyang Xu, Xiao-Jun Wu.<br />
  "A Survey for Deep RGBT Tracking." ArXiv (2022).
  [[paper](https://arxiv.org/abs/2201.09296)] 

- Jinyu Yang, Zhe Li, Song Yan, Feng Zheng, Aleš Leonardis, Joni-Kristian Kämäräinen, Ling Shao.<br />
  "RGBD Object Tracking: An In-depth Review." ArXiv (2022).
  [[paper](https://arxiv.org/abs/2203.14134)] 

- Chenglong Li, Andong Lu, Lei Liu, Jin Tang.<br />
  "Multi-modal visual tracking: a survey. 多模态视觉跟踪方法综述" Journal of Image and Graphics.中国图象图形学报 (2023).
  [[paper](http://www.cjig.cn/html/2023/1/20230103.htm)] 

- Ou Zhou, Ying Ge, Zhang Dawei, and Zheng Zhonglong.<br />
  "A Survey of RGB-Depth Object Tracking. RGB-D 目标跟踪综述" Journal of Computer-Aided Design & Computer Graphics. 计算机辅助设计与图形学学报 (2024).
  [[paper](https://www.jcad.cn/cn/article/doi/10.3724/SP.J.1089.null.2023-00537)] 

- Zhang, ZhiHao and Wang, Jun and Zang, Zhuli and Jin, Lei and Li, Shengjie and Wu, Hao and Zhao, Jian and Bo, Zhang.<br />
  "Review and Analysis of RGBT Single Object Tracking Methods: A Fusion Perspective." ACM Transactions on Multimedia Computing, Communications and Applications (2024).
  [[paper](https://dl.acm.org/doi/pdf/10.1145/3651308)] 

- **MV-RGBT & MoETrack:** Zhangyong Tang, Tianyang Xu, Zhenhua Feng, Xuefeng Zhu, He Wang, Pengcheng Shao, Chunyang Cheng, Xiao-Jun Wu, Muhammad Awais, Sara Atito, Josef Kittler.<br />
  "Revisiting RGBT Tracking Benchmarks from the Perspective of Modality Validity: A New Benchmark, Problem, and Method." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2405.00168)] 
  [[code](https://github.com/Zhangyong-Tang/MoETrack)]

- Xingchen Zhang and Ping Ye and Henry Leung and Ke Gong and Gang Xiao.<br />
  "Object fusion tracking based on visible and infrared images: A comprehensive review." Information Fusion (2024).
  [[paper](https://www.sciencedirect.com/science/article/pii/S1566253520302657)] 

- Mingzheng Feng and Jianbo Su.<br />
  "RGBT tracking: A comprehensive review." Information Fusion (2024).
  [[paper](https://www.sciencedirect.com/science/article/pii/S1566253524002707)] 

- Zhang, Haiping and Yuan, Di and Shu, Xiu and Li, Zhihui and Liu, Qiao and Chang, Xiaojun and He, Zhenyu and Shi, Guangming.<br />
  "A Comprehensive Review of RGBT Tracking." IEEE TIM (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10616144)] 

- Mengmeng Wang, Teli Ma, Shuo Xin, Xiaojun Hou, Jiazheng Xing, Guang Dai, Jingdong Wang, Yong Liu.<br />
  "Visual Object Tracking across Diverse Data Modalities: A Review." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2412.09991)] 

## Vision-Language Tracking
### Datasets
| Dataset | Pub. & Date  | WebSite | Introduction |
|:-----:|:-----:|:-----:|:-----:|
|  [OTB99-L](https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Tracking_by_Natural_CVPR_2017_paper.pdf)   |  CVPR-2017  |  [OTB99-L](https://github.com/QUVA-Lab/lang-tracker)  |   99 videos  |  
|  [LaSOT](https://arxiv.org/abs/2009.03465)   |   CVPR-2019   |  [LaSOT](https://github.com/HengLan/LaSOT_Evaluation_Toolkit)  | 1400 videos  |  
|  [LaSOT_EXT](https://arxiv.org/pdf/1809.07845.pdf)   |   IJCV-2021   |  [LaSOT_EXT](https://github.com/HengLan/LaSOT_Evaluation_Toolkit)  |  150 videos  |  
|  [TNL2K](https://arxiv.org/pdf/2103.16746.pdf)   |   CVPR-2021  |  [TNL2K](https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit)  |  2000 videos  |  
|  [WebUAV-3M](https://arxiv.org/abs/2201.07425)   |   TPAMI-2023   |  [WebUAV-3M](https://github.com/983632847/WebUAV-3M)  |  4500 videos, 3.3 million frames, UAV tracking, vision-language-audio |  
|  [MGIT](https://huuuuusy.github.io/files/MGIT.pdf)   |   NeurIPS-2023   |  [MGIT](http://videocube.aitestunion.com/)  |  150 long video sequences, 2.03 million frames,  three semantic grains (i.e., action, activity, and story)  |  
|  [VastTrack](https://arxiv.org/abs/2403.03493)   |   NeurIPS-2024   |  [VastTrack](https://github.com/HengLan/VastTrack)  |  50,610 video sequences, 4.2 million frames, 2,115 classes  |  
|  [WebUOT-1M](https://arxiv.org/abs/2405.19818)   |   NeurIPS-2024   |  [WebUOT-1M](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/WebUOT-1M)  |  The first million-scale underwater object tracking dataset contains 1,500 video sequences, 1.1 million frames |  
|  [ElysiumTrack-1M](https://arxiv.org/abs/2403.16558)   |   ECCV-2024   |  [ElysiumTrack-1M](https://github.com/Hon-Wong/Elysium)  |   A large-scale dataset that supports three tasks: single object tracking, reference single object tracking, and video reference expression generation, with 1.27 million videos |  
|  [VLT-MI](https://arxiv.org/abs/2409.08887)   |   arXiv-2024   | [VLT-MI](http://videocube.aitestunion.com/) |  A dataset for multi-round, multi-modal interaction, with 3,619 videos. |  
|  [DTVLT](https://arxiv.org/abs/2410.02492)   |   arXiv-2024   | [DTVLT](http://videocube.aitestunion.com/) |  A multi-modal diverse text benchmark for visual language tracking (RGBL Tracking). |  
|  [SemTrack](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03555.pdf)   |   ECCV-2024   | [SemTrack](https://forms.office.com/Pages/ResponsePage.aspx?id=drd2NJDpck-5UGJImDFiPQJNzw6AhuZDkzEViiWzJltUNjhKM01KWjhXN0FBNjcxNVBZQk03VVFHQi4u) |   A large-scale dataset comprising 6.7 million frames from 6,961 videos, capturing the semantic trajectory of targets across 52 interaction classes and 115 object classes. |
|  [UW-COT220](https://arxiv.org/abs/2409.16902)   |   arXiv-2025   | [UW-COT220](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/UW-COT220) |  The first multimodal underwater camouﬂaged object tracking dataset with 220 videos. |  

### Papers
#### 2025
- **SIEVL-Track:** Li, Ning and Zhong, Bineng and Liang, Qihua and Mo, Zhiyi and Nong, Jian and Song, Shuxiang.<br />
  "SIEVL-Track: Exploring Semantic Information Enhancement for Visual-Language Object Tracking." TCSVT (2025).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10845881)] 

- **UW-COT220 & VL-SAM2:** Chunhui Zhang, Li Liu, Guanjie Huang, Hao Wen, Xi Zhou, Yanfeng Wang.<br />
  "Towards Underwater Camouflaged Object Tracking: Benchmark and Baselines." ArXiv (2025).
  [[paper](https://arxiv.org/abs/2409.16902)]
  [[ResearchGate](https://www.researchgate.net/publication/388189638_Towards_Underwater_Camouflaged_Object_Tracking_Benchmark_and_Baselines)]
  [[知乎](https://zhuanlan.zhihu.com/p/19568835256)]
  [[project](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/UW-COT220)]
  
- **CTVLT:** X. Feng, D. Zhang, S. Hu, X. Li, M. Wu, J. Zhang, X. Chen, K. Huang.<br />
  "Enhancing Vision-Language Tracking by Effectively Converting Textual Cues into Visual Cues." ICASSP  (2025).
  [[paper](https://arxiv.org/abs/2412.19648)] 
  [[code](https://github.com/XiaokunFeng/CTVLT)]
  
#### 2024
- **JLPT:** Weng, ZhiMin and Zhang, JinPu and Wang, YueHuan.<br />
  "Joint Language Prompt and Object Tracking." ICME (2024).
  [[paper](https://ieeexplore.ieee.org/document/10687451)] 

- **CPIPTrack:** Zhu, Hong and Lu, Qingyang and Xue, Lei and Zhang, Pingping and Yuan, Guanglin.<br />
  "Vision-Language Tracking With CLIP and Interactive Prompt Learning." TITS (2024).
  [[paper](https://ieeexplore.ieee.org/document/10817474)] 

- **DMITrack:** Zhiyi Mo, Guangtong Zhang, Jian Nong, Bineng Zhong, Zhi Li.<br />
  "Dual-stream Multi-modal Interactive Vision-language Tracking." MMAsia (2024).
[[paper](https://dl.acm.org/doi/10.1145/3696409.3700220)] 

- **PJVLT:** Liang, Yanjie and Wu, Qiangqiang and Cheng, Lin and Xia, Changqun and Li, Jia.<br />
  "Progressive Semantic-Visual Alignment and Refinement for Vision-Language Tracking." TCSVT (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10807368)] 

- **MugTracker:** Zhu, Hong and Zhang, Pingping and Xue, Lei and Yuan, Guanglin.<br />
  "Multi-modal Understanding and Generation for Object Tracking." TCSVT (2024).
  [[paper](https://ieeexplore.ieee.org/document/10777574)] 

- **CogVLM-Track:** Xuexin Liu, Zhuojun Zou & Jie Hao.<br />
  "Adaptive Text Feature Updating for Visual-Language Tracking." ICPR (2024).
  [[paper](https://link.springer.com/chapter/10.1007/978-3-031-78110-0_24)] 

- **VLTVerse:** Xuchen Li, Shiyu Hu, Xiaokun Feng, Dailing Zhang, Meiqi Wu, Jing Zhang, Kaiqi Huang.<br />
  "How Texts Help? A Fine-grained Evaluation to Reveal the Role of Language in Vision-Language Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2411.15600)] 
  [[project](http://metaverse.aitestunion.com/)]

- **MambaVLT:** Xinqi Liu, Li Zhou, Zikun Zhou, Jianqiu Chen, Zhenyu He.<br />
  "MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2411.15459)] 
  
- Li, Hengyou and Liu, Xinyan and Li, Guorong and Wang, Shuhui and Qing, Laiyun and Huang, Qingming.<br />
  "Boost Tracking by Natural Language With Prompt-Guided Grounding." TITS (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10756281)] 

- **ChatTracker:** Yiming Sun, Fan Yu, Shaoxiang Chen, Yu Zhang, Junwei Huang, Chenhui Li, Yang Li, Changbo Wang.<br />
  "ChatTracker: Enhancing Visual Tracking Performance via Chatting with Multimodal Large Language Model." NeurIPS (2024).
  [[paper](https://arxiv.org/abs/2411.01756)] 

- **SemTrack:** Wang, Pengfei and Hui, Xiaofei and Wu, Jing and Yang, Zile and Ong, Kian Eng and Zhao, Xinge and Lu, Beijia and Huang, Dezhao and Ling, Evan and Chen, Weiling and Ma, Keng Teck and Hur, Minhoe and Liu, Jun.<br />
  "SemTrack: A Large-scale Dataset for Semantic Tracking in the Wild." ECCV (2024).
  [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03555.pdf)] 
  [[project](https://sutdcv.github.io/SemTrack/)]

- **MemVLT**: Xiaokun Feng, Xuchen Li, Shiyu Hu, Dailing Zhang, Meiqi Wu, Jing Zhang, Xiaotang Chen, Kaiqi Huang.<br />
  "MemVLT: Visual-Language Tracking with Adaptive Memory-based Prompts." NeurIPS (2024).
  [[paper](https://openreview.net/pdf?id=ZK1CZXKgG5)] 
  [[code](https://github.com/XiaokunFeng/MemVLT)]
  
- **DTVLT:** Xuchen Li, Shiyu Hu, Xiaokun Feng, Dailing Zhang, Meiqi Wu, Jing Zhang, Kaiqi Huang.<br />
  "DTVLT: A Multi-modal Diverse Text Benchmark for Visual Language Tracking Based on LLM." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2410.02492)] 
  [[project](http://videocube.aitestunion.com/)]

- **WebUOT-1M:** Chunhui Zhang, Li Liu, Guanjie Huang, Hao Wen, Xi Zhou, Yanfeng Wang.<br />
  "WebUOT-1M: Advancing Deep Underwater Object Tracking with A Million-Scale Benchmark." NeurIPS (2024).
  [[paper](https://arxiv.org/abs/2405.19818)] 
  [[project](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/WebUOT-1M)]
  
- **MambaTrack:** Chunhui Zhang, Li Liu, Hao Wen, Xi Zhou, Yanfeng Wang.<br />
  "MambaTrack: Exploiting Dual-Enhancement for Night UAV Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2411.15761)]
  
- **ElysiumTrack-1M:** Han Wang, Yanjie Wang, Yongjie Ye, Yuxiang Nie, Can Huang.<br />
  "Elysium: Exploring Object-level Perception in Videos via MLLM." ECCV (2024).
  [[paper](https://arxiv.org/abs/2403.16558)] 
  [[code](https://github.com/Hon-Wong/Elysium)]

- **VLT-MI:** Xuchen Li, Shiyu Hu, Xiaokun Feng, Dailing Zhang, Meiqi Wu, Jing Zhang, Kaiqi Huang.<br />
  "Visual Language Tracking with Multi-modal Interaction: A Robust Benchmark." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2409.08887)]
  [[project](http://videocube.aitestunion.com/)]

- **VastTrack:** Liang Peng, Junyuan Gao, Xinran Liu, Weihong Li, Shaohua Dong, Zhipeng Zhang, Heng Fan, Libo Zhang.<br />
  "VastTrack: Vast Category Visual Object Tracking." NeurIPS (2024).
  [[paper](https://arxiv.org/abs/2403.03493)] 
  [[project](https://github.com/HengLan/VastTrack)]

- **DMTrack:** Guangtong Zhang, Bineng Zhong, Qihua Liang, Zhiyi Mo, Shuxiang Song.<br />
  "Diffusion Mask-Driven Visual-language Tracking." IJCAI (2024).
  [[paper](https://www.ijcai.org/proceedings/2024/0183.pdf)] 

- **ATTracker:** Jiawei Ge, Jiuxin Cao, Xuelin Zhu, Xinyu Zhang, Chang Liu, Kun Wang, Bo Liu.<br />
  "Consistencies are All You Need for Semi-supervised Vision-Language Tracking." ACM MM (2024).
  [[paper](https://openreview.net/pdf?id=jLJ3htNxVX)] 

- **ALTracker:** Zikai Song, Ying Tang, Run Luo, Lintao Ma, Junqing Yu, Yi-Ping Phoebe Chen, Wei Yang.<br />
  "Autogenic Language Embedding for Coherent Point Tracking." ACM MM (2024).
  [[paper](https://arxiv.org/abs/2407.20730)] 
  [[code](https://github.com/SkyeSong38/ALTrack)]

- **Elysium:** Han Wang, Yanjie Wang, Yongjie Ye, Yuxiang Nie, Can Huang.<br />
  "Elysium: Exploring Object-level Perception in Videos via MLLM." ECCV (2024).
  [[paper](https://arxiv.org/abs/2403.16558)] 
  [[code](https://github.com/Hon-Wong/Elysium)]

- **Tapall.ai:** Mingqi Gao, Jingnan Luo, Jinyu Yang, Jungong Han, Feng Zheng.<br />
  "1st Place Solution for MeViS Track in CVPR 2024 PVUW Workshop: Motion Expression guided Video Segmentation." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2406.07043)] 
  [[code](https://github.com/Tapall-AI/MeViS_Track_Solution_2024)]

- **DTLLM-VLT:** Xuchen Li, Xiaokun Feng, Shiyu Hu, Meiqi Wu, Dailing Zhang, Jing Zhang, Kaiqi Huang.<br />
  "DTLLM-VLT: Diverse Text Generation for Visual Language Tracking Based on LLM." CVPRW (2024).
  [[paper](https://arxiv.org/abs/2405.12139)]
  [[project](http://videocube.aitestunion.com/)]

- **UVLTrack:** Yinchao Ma, Yuyang Tang, Wenfei Yang, Tianzhu Zhang, Jinpeng Zhang, Mengxue Kang.<br />
  "Unifying Visual and Vision-Language Tracking via Contrastive Learning." AAAI (2024).
  [[paper](https://arxiv.org/abs/2401.11228)] 
  [[code](https://github.com/OpenSpaceAI/UVLTrack)]

- **QueryNLT:** Yanyan Shao, Shuting He, Qi Ye, Yuchao Feng, Wenhan Luo, Jiming Chen.<br />
  "Context-Aware Integration of Language and Visual References for Natural Language Tracking." CVPR (2024).
  [[paper](https://arxiv.org/abs/2403.19975)] 
  [[code](https://github.com/twotwo2/QueryNLT)]

- **OSDT:** Guangtong Zhang, Bineng Zhong, Qihua Liang, Zhiyi Mo, Ning Li, Shuxiang Song.<br />
  "One-Stream Stepwise Decreasing for Vision-Language Tracking." TCSVT (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10510485)] 

- **TTCTrack:** Zhongjie Mao; Yucheng Wang; Xi Chen; Jia Yan.<br />
  "Textual Tokens Classification for Multi-Modal Alignment in Vision-Language Tracking." ICASSP (2024).
  [[paper](https://ieeexplore.ieee.org/document/10446122)] 

- **OneTracker:** Lingyi Hong, Shilin Yan, Renrui Zhang, Wanyun Li, Xinyu Zhou, Pinxue Guo, Kaixun Jiang, Yiting Cheng, Jinglun Li, Zhaoyu Chen, Wenqiang Zhang.<br />
  "OneTracker: Unifying Visual Object Tracking with Foundation Models and Efficient Tuning." CVPR (2024).
  [[paper](https://arxiv.org/pdf/2403.09634.pdf)] 

- **MMTrack:** Zheng, Yaozong and Zhong, Bineng and Liang, Qihua and Li, Guorong and Ji, Rongrong and Li, Xianxian.<br />
  "Toward Unified Token Learning for Vision-Language Tracking." TCSVT (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10208210)]
  [[code](https://github.com/Azong-HQU/MMTrack)]

- Ping Ye, Gang Xiao, Jun Liu .<br />
  "Multimodal Features Alignment for Vision–Language Object Tracking." Remote Sensing (2024).
  [[paper](https://www.mdpi.com/2072-4292/16/7/1168)] 

- **VLT_OST:** Mingzhe Guo, Zhipeng Zhang, Liping Jing, Haibin Ling, Heng Fan.<br />
  "Divert More Attention to Vision-Language Object Tracking." TPAMI (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10547435)] 
  [[code](https://github.com/JudasDie/SOTS)]

- **SATracker:** Jiawei Ge, Xiangmei Chen, Jiuxin Cao, Xuelin Zhu, Weijia Liu, Bo Liu.<br />
  "Beyond Visual Cues: Synchronously Exploring Target-Centric Semantics for Vision-Language Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/pdf/2311.17085.pdf)]

- **VLFSE:** Fuchao Yang, Mingkai Jiang, Qiaohong Hao, Xiaolei Zhao, Qinghe Feng.<br />
  "VLFSE: Enhancing visual tracking through visual language fusion and state update evaluator." Machine Learning with Applications (2024).
  [[paper](https://www.sciencedirect.com/science/article/pii/S2666827024000641)] 

  
#### 2023
- **WebUAV-3M:** Chunhui Zhang, Guanjie Huang, Li Liu, Shan Huang, Yinan Yang, Xiang Wan, Shiming Ge, Dacheng Tao.<br />
  "WebUAV-3M: A Benchmark for Unveiling the Power of Million-Scale Deep UAV Tracking." TPAMI (2023).
  [[paper](https://arxiv.org/abs/2201.07425)] 
  [[project](https://github.com/983632847/WebUAV-3M)]
  
- **All in One:** Chunhui Zhang, Xin Sun, Li Liu, Yiqian Yang, Qiong Liu, Xi Zhou, Yanfeng Wang.<br />
  "All in One: Exploring Unified Vision-Language Tracking with Multi-Modal Alignment." ACM MM (2023).
  [[paper](https://arxiv.org/abs/2307.03373)] 
  [[code](https://github.com/983632847/All-in-One)]

- **CiteTracker:** Xin Li, Yuqing Huang, Zhenyu He, Yaowei Wang, Huchuan Lu, Ming-Hsuan Yang.<br />
  "CiteTracker: Correlating Image and Text for Visual Tracking." ICCV (2023).
  [[paper](https://arxiv.org/abs/2308.11322)] 
  [[code]( https://github.com/NorahGreen/CiteTracker)]

- **JointNLT:** Li Zhou, Zikun Zhou, Kaige Mao, Zhenyu He.<br />
  "Joint Visual Grounding and Tracking with Natural Language Specifcation." CVPR (2023).
  [[paper](https://arxiv.org/abs/2303.12027#:~:text=Tracking%20by%20natural%20language%20specification%20aims%20to%20locate,tracking%20model%20to%20implement%20these%20two%20steps%2C%20respectively.)] 
  [[code](https://github.com/lizhou-cs/JointNLT)]

- **MGIT:** Hu, Shiyu and Zhang, Dailing and meiqi, wu and Feng, Xiaokun and Li, Xuchen and Zhao, Xin and Huang, Kaiqi.<br />
  "A Multi-modal Global Instance Tracking Benchmark (MGIT): Better Locating Target in Complex Spatio-temporal and Causal Relationship." NeurIPS (2023).
  [[paper](https://huuuuusy.github.io/files/MGIT.pdf)] 
  [[project](http://videocube.aitestunion.com/)]

- **DecoupleTNL:** Ma, Ding and Wu, Xiangqian.<br />
  "Tracking by Natural Language Specification with Long Short-term Context Decoupling." ICCV (2023).
  [[paper](https://ieeexplore.ieee.org/document/10378598/references#references)] 

- Haojie Zhao, Xiao Wang, Dong Wang, Huchuan Lu, Xiang Ruan.<br />
  "Transformer vision-language tracking via proxy token guided
  cross-modal fusion." PRL (2023).
  [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865523000545)] 

- **OVLM:** Zhang, Huanlong and Wang, Jingchao and Zhang, Jianwei and Zhang, Tianzhu and Zhong, Bineng.<br />
  "One-Stream Vision-Language Memory Network for Object Tracking." TMM (2023).
  [[paper](https://ieeexplore.ieee.org/document/10149530)]
  [[code](https://github.com/wjc0602/OVLM)]
  
- **VLATrack:** Zuo, Jixiang and Wu, Tao and Shi, Meiping and Liu, Xueyan and Zhao, Xijun.<br />
  "Multi-Modal Object Tracking with Vision-Language Adaptive Fusion and Alignment." RICAI (2023).
  [[paper](https://ieeexplore.ieee.org/document/10489325)] 

#### 2022

- **VLT_TT:** Mingzhe Guo, Zhipeng Zhang, Heng Fan, Liping Jing.<br />
  "Divert More Attention to Vision-Language Tracking." NeurIPS (2022).
  [[paper](https://arxiv.org/abs/2207.01076)] 
  [[code](https://github.com/JudasDie/SOTS)]


- **AdaRS:** Li, Yihao and Yu, Jun and Cai, Zhongpeng and Pan, Yuwen.<br />
  "Cross-modal Target Retrieval for Tracking by Natural Language." CVPR Workshops (2022).
  [[paper](https://ieeexplore.ieee.org/document/9857151)] 

#### 2021
- **TNL2K:** Wang, Xiao and Shu, Xiujun and Zhang, Zhipeng and Jiang, Bo and Wang, Yaowei and Tian, Yonghong and Wu, Feng.<br />
  "Towards More Flexible and Accurate Object Tracking with Natural Language: Algorithms and Benchmark." CVPR (2021).
  [[paper](https://arxiv.org/abs/1809.07845)] 
  [[project](https://sites.google.com/view/langtrackbenchmark/)]

- **LaSOT_EXT:** Heng Fan, Hexin Bai, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Harshit, Mingzhen Huang, Juehuan Liu, Yong Xu, Chunyuan Liao, Lin Yuan, Haibin Ling.<br />
  "LaSOT: A High-quality Large-scale Single Object Tracking Benchmark." IJCV (2021).
  [[paper](https://arxiv.org/abs/2009.03465)] 
  [[project](https://github.com/HengLan/LaSOT_Evaluation_Toolkit)]

- **SNLT:** Qi Feng, Vitaly Ablavsky, Qinxun Bai, Stan Sclaroff.<br />
  "Siamese Natural Language Tracker: Tracking by Natural Language Descriptions with Siamese Trackers." CVPR  (2021).
  [[paper](https://arxiv.org/abs/1912.02048)] 
  [[code](https://github.com/fredfung007/snlt)]
  
#### 2019
- **LaSOT:** Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, Haibin Ling.<br />
  "LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking." CVPR (2021).
  [[paper](https://arxiv.org/abs/1809.07845)] 
  [[project](http://vision.cs.stonybrook.edu/~lasot/)]

#### 2017
- **OTB99-L:** Zhenyang Li, Ran Tao, Efstratios Gavves, Cees G. M. Snoek, Arnold W.M. Smeulders.<br />
  "Tracking by Natural Language Specification." CVPR (2017).
  [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Tracking_by_Natural_CVPR_2017_paper.pdf)] 
  [[project](https://github.com/QUVA-Lab/lang-tracker)]



## RGBE Tracking
### Datasets
| Dataset | Pub. & Date  | WebSite | Introduction |
|:-----:|:-----:|:-----:|:-----:|
|  [FE108](https://arxiv.org/abs/2109.09052)   |  ICCV-2021   |  [FE108](https://zhangjiqing.com/dataset/)  |  108 event videos  |  
|  [COESOT](https://arxiv.org/abs/2211.11010)   |   arXiv-2022   |  [COESOT](https://github.com/Event-AHU/COESOT)  |  1354 RGB-event video pairs  |  
|  [VisEvent](https://arxiv.org/abs/2108.05015)   |   TC-2023   |  [VisEvent](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)  |  820 RGB-event video pairs |  
|  [EventVOT](https://arxiv.org/abs/2309.14611)   |   CVPR-2024   |  [EventVOT](https://github.com/Event-AHU/EventVOT_Benchmark)  |  1141 event videos  |  
|  [CRSOT](https://arxiv.org/abs/2401.02826)   |   arXiv-2024   |  [CRSOT](https://github.com/Event-AHU/Cross_Resolution_SOT)  |   1030 RGB-event video pairs |  
|  [FELT](https://arxiv.org/pdf/2403.05839.pdf)   |   arXiv-2024   |  [FELT](https://github.com/Event-AHU/FELT_SOT_Benchmark)  |  742 RGB-event video pairs  |  
|  [MEVDT](https://arxiv.org/abs/2407.20446)   |   arXiv-2024   |  [MEVDT](https://doi.org/10.7302/d5k3-9150)  |  63 multimodal sequences with 13k images, 5M events, 10k object labels and 85 trajectories  |  


### Papers
#### 2025
- **HDETrack V2:** Shiao Wang, Xiao Wang, Chao Wang, Liye Jin, Lin Zhu, Bo Jiang, Yonghong Tian, Jin Tang.<br />
  "Event Stream-based Visual Object Tracking: HDETrack V2 and A High-Definition Benchmark." ArXiv (2025).
  [[paper](https://arxiv.org/abs/2502.05574)] 
  [[code](https://github.com/Event-AHU/EventVOT_Benchmark)]

#### 2024
- **CSAM:** Tianlu Zhang, Kurt Debattista, Qiang Zhang, Guiguang Ding, Jungong Han.<br />
  "Revisiting motion information for RGB-Event tracking with MOT philosophy." NeurIPS (2024).
  [[paper](https://openreview.net/forum?id=bzGAELYOyL)] 

- **GS-EVT:** Tao Liu, Runze Yuan, Yi'ang Ju, Xun Xu, Jiaqi Yang, Xiangting Meng, Xavier Lagorce, Laurent Kneip.<br />
  "GS-EVT: Cross-Modal Event Camera Tracking based on Gaussian Splatting." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2409.19228)] 

- **DS-MESA:** Pengcheng Shao, Tianyang Xu, Xuefeng Zhu, Xiaojun Wu, Josef Kittler.<br />
  "Dynamic Subframe Splitting and Spatio-Temporal Motion Entangled Sparse Attention for RGB-E Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2409.17560)] 

- **BlinkTrack:** Yichen Shen, Yijin Li, Shuo Chen, Guanglin Li, Zhaoyang Huang, Hujun Bao, Zhaopeng Cui, Guofeng Zhang.<br />
  "BlinkTrack: Feature Tracking over 100 FPS via Events and Images." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2409.17981)] 

- **FE-TAP:** Jiaxiong Liu, Bo Wang, Zhen Tan, Jinpu Zhang, Hui Shen, Dewen Hu.<br />
  "Tracking Any Point with Frame-Event Fusion Network at High Frame Rate." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2409.11953)] 
  [[code](https://github.com/ljx1002/FE-TAP)]

- **MambaEVT:** Xiao Wang, Chao wang, Shiao Wang, Xixi Wang, Zhicheng Zhao, Lin Zhu, Bo Jiang.<br />
  "MambaEVT: Event Stream based Visual Object Tracking using State Space Model." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2408.10487)] 
  [[code](https://github.com/Event-AHU/MambaEVT)]

- **eMoE-Tracker:** Yucheng Chen, Lin Wang.<br />
  "eMoE-Tracker: Environmental MoE-based Transformer for Robust Event-guided Object Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2406.20024)] 
  [[code](https://vlislab22.github.io/eMoE-Tracker/)]

- **ED-DCFNet:** Raz Ramon, Hadar Cohen-Duwek, Elishai Ezra Tsur.<br />
  "ED-DCFNet: An Unsupervised Encoder-decoder Neural Model for Event-driven Feature Extraction and Object Tracking." CVPRW (2024).
  [[paper](https://openaccess.thecvf.com/content/CVPR2024W/EVW/papers/Ramon_ED-DCFNet_An_Unsupervised_Encoder-decoder_Neural_Model_for_Event-driven_Feature_Extraction_CVPRW_2024_paper.pdf)] 
  [[code](https://github.com/NBELab/UnsupervisedTracking)]
  
- **Mamba-FETrack:** Ju Huang, Shiao Wang, Shuai Wang, Zhe Wu, Xiao Wang, Bo Jiang.<br />
  "Mamba-FETrack: Frame-Event Tracking via State Space Model." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2404.18174)] 
  [[code](https://github.com/Event-AHU/Mamba_FETrack)]

- **AMTTrack:** Xiao Wang, Ju Huang, Shiao Wang, Chuanming Tang, Bo Jiang, Yonghong Tian, Jin Tang, Bin Luo.<br />
  "Long-term Frame-Event Visual Tracking: Benchmark Dataset and Baseline." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2401.02826)] 
  [[code](https://github.com/Event-AHU/FELT_SOT_Benchmark)]

- **TENet:** Pengcheng Shao, Tianyang Xu, Zhangyong Tang, Linze Li, Xiao-Jun Wu, Josef Kittler.<br />
  "TENet: Targetness Entanglement Incorporating with Multi-Scale Pooling and Mutually-Guided Fusion for RGB-E Object Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2405.05004)] 
  [[code](https://github.com/SSSpc333/TENet)]

- **HDETrack:** Xiao Wang, Shiao Wang, Chuanming Tang, Lin Zhu, Bo Jiang, Yonghong Tian, Jin Tang.<br />
  "Event Stream-based Visual Object Tracking: A High-Resolution Benchmark Dataset and A Novel Baseline." CVPR (2024).
  [[paper](https://arxiv.org/abs/2309.14611)] 
  [[code](https://github.com/Event-AHU/EventVOT_Benchmark)]

- Yabin Zhu, Xiao Wang, Chenglong Li, Bo Jiang, Lin Zhu, Zhixiang Huang, Yonghong Tian, Jin Tang.<br />
  "CRSOT: Cross-Resolution Object Tracking using Unaligned Frame and Event Cameras." ArXiv (2024).
  [[paper](https://arxiv.org/pdf/2403.05839.pdf)] 
  [[code](https://github.com/Event-AHU/FELT_SOT_Benchmark)]

- **CDFI:** Jiqing Zhang, Xin Yang, Yingkai Fu, Xiaopeng Wei, Baocai Yin, Bo Dong.<br />
  "Object Tracking by Jointly Exploiting Frame and Event Domain." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2109.09052)]

- **MMHT:** Hongze Sun, Rui Liu, Wuque Cai, Jun Wang, Yue Wang, Huajin Tang, Yan Cui, Dezhong Yao, Daqing Guo.<br />
  "Reliable Object Tracking by Multimodal Hybrid Feature Extraction and Transformer-Based Fusion." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2405.17903)] 


#### 2023
- Zhiyu Zhu, Junhui Hou, Dapeng Oliver Wu.<br />
  "Cross-modal Orthogonal High-rank Augmentation for RGB-Event Transformer-trackers." ICCV (2023).
  [[paper](https://arxiv.org/abs/2307.04129)] 
  [[code](https://github.com/ZHU-Zhiyu/High-Rank_RGB-Event_Tracker)]

- **AFNet:** Jiqing Zhang, Yuanchen Wang, Wenxi Liu, Meng Li, Jinpeng Bai, Baocai Yin, Xin Yang.<br />
  "Frame-Event Alignment and Fusion Network for High Frame Rate Tracking." CVPR (2023).
  [[paper](https://arxiv.org/abs/2305.15688)] 
  [[code](https://github.com/Jee-King/AFNet)]

- **RT-MDNet:** Xiao Wang, Jianing Li, Lin Zhu, Zhipeng Zhang, Zhe Chen, Xin Li, Yaowei Wang, Yonghong Tian, Feng Wu.<br />
  "VisEvent: Reliable Object Tracking via Collaboration of Frame and Event Flows." TC (2023).
  [[paper](https://arxiv.org/abs/2108.05015)] 
  [[code](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)]

#### 2022
- **Event-tracking:** Zhiyu Zhu, Junhui Hou, Xianqiang Lyu.<br />
  "Learning Graph-embedded Key-event Back-tracing for Object Tracking in Event Clouds." NeurIPS (2022).
  [[paper](https://dl.acm.org/doi/10.5555/3600270.3600812)] 
  [[code](https://github.com/ZHU-Zhiyu/Event-tracking)]

- **STNet:** Jiqing Zhang, Bo Dong, Haiwei Zhang, Jianchuan Ding, Felix Heide, Baocai Yin, Xin Yang.<br />
  "Spiking Transformers for Event-based Single Object Tracking." CVPR (2022).
  [[paper](https://ieeexplore.ieee.org/document/9879994)] 
  [[code](https://github.com/Jee-King/CVPR2022_STNet)]

- **CEUTrack:** Chuanming Tang, Xiao Wang, Ju Huang, Bo Jiang, Lin Zhu, Jianlin Zhang, Yaowei Wang, Yonghong Tian.<br />
  "Revisiting Color-Event based Tracking: A Unified Network, Dataset, and Metric." ArXiv (2022).
  [[paper](https://arxiv.org/abs/2211.11010)] 
  [[code](https://github.com/Event-AHU/COESOT)]


#### 2021

- **CFE:** Jiqing Zhang, Kai Zhao, Bo Dong, Yingkai Fu, Yuxin Wang, Xin Yang, Baocai Yin.<br />
  "Multi-domain Collaborative Feature Representation for Robust Visual Object Tracking." The Visual Computer (2021).
  [[paper](https://arxiv.org/abs/2108.04521)]


## RGBD Tracking
### Datasets
| Dataset | Pub. & Date  | WebSite | Introduction |
|:-----:|:-----:|:-----:|:-----:|
|  [PTB](https://vision.princeton.edu/projects/2013/tracking/paper.pdf)   |   ICCV-2013   |  [PTB](https://tracking.cs.princeton.edu/index.html)  |  100 sequences  |  
|  [STC](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8026575)   |   TC-2018   |  [STC](https://beardatashare.bham.ac.uk/dl/fiVnhJRjkyNN8QjSAoiGSiBY/RGBDdataset.zip)  |  36 sequences  |  
|  [CDTB](https://arxiv.org/pdf/1907.00618.pdf)   |   ICCV-2019  |  [CDTB](https://www.votchallenge.net/vot2019/dataset.html)  |  80 sequences  |  
|  [VOT-RGBD 2019/2020/2021](https://openaccess.thecvf.com/content_ICCVW_2019/papers/VOT/Kristan_The_Seventh_Visual_Object_Tracking_VOT2019_Challenge_Results_ICCVW_2019_paper.pdf)   |   ICCVW-2019  |  [VOT-RGBD 2019](https://votchallenge.net/vot2019/dataset.html)  | VOT-RGBD 2019, 2020, and 2021 are based on CDTB |    
|  [DepthTrack](https://arxiv.org/abs/2108.13962)   |   ICCV-2021   |  [DepthTrack](http://doi.org/10.5281/zenodo.4716441)  |  200 sequences  |  
|  [VOT-RGBD 2022](https://link.springer.com/chapter/10.1007/978-3-031-25085-9_25)   |   ECCVW-2022  |  [VOT-RGBD 2022](https://votchallenge.net/vot2022/dataset.html)  | VOT-RGBD 2022 is based on CDTB and DepthTrack |  
|  [RGBD1K](https://arxiv.org/abs/2208.09787)   |   AAAI-2023   |  [RGBD1K](https://github.com/xuefeng-zhu5/RGBD1K)  |   1,050 sequences, 2.5M frames |  
|  [DTTD](https://arxiv.org/abs/2302.05991)   |   CVPR Workshops-2023   |  [DTTD](https://github.com/augcog/DTTDv1)  | 103 scenes, 55691 frames   |  
|  [ARKitTrack](https://arxiv.org/abs/2303.13885)   |   CVPR-2023   |  [ARKitTrack](https://arkittrack.github.io)  |  300 RGB-D sequences, 455 targets, 229.7K video frames  |  

### Papers
#### 2024
- **DAMT:** Yifan Pan, Tianyang Xu, Xue-Feng Zhu, Xiaoqing Luo, Xiao-Jun Wu & Josef Kittler .<br />
  "Learning Explicit Modulation Vectors for Disentangled Transformer Attention-Based RGB-D Visual Tracking." ICPR (2024).
  [[paper](https://link.springer.com/chapter/10.1007/978-3-031-78444-6_22)] 

- **3DPT:** Bocen Li, Yunzhi Zhuge, Shan Jiang, Lijun Wang, Yifan Wang, Huchuan Lu.<br />
  "3D Prompt Learning for RGB-D Tracking." ACCV (2024).
  [[paper](https://openaccess.thecvf.com/content/ACCV2024/html/Li_3D_Prompt_Learning_for_RGB-D_Tracking_ACCV_2024_paper.html)] 

- **UBPT:** Ou, Zhou and Zhang, Dawei and Ying, Ge and Zheng, Zhonglong.<br />
  "UBPT: Unidirectional and Bidirectional Prompts for RGBD Tracking." IEEE Sensors Journal (2024).
  [[paper](https://ieeexplore.ieee.org/document/10706817)] 

- **L2FIG-Tracker:** Jintao Su, Ye Liu, Shitao Song .<br />
  "L2FIG-Tracker: L2-Norm Based Fusion with Illumination Guidance for RGB-D Object Tracking." PRCV (2024).
  [[paper](https://link.springer.com/chapter/10.1007/978-981-97-8493-6_10)] 

- **Depth Attention:** Yu Liu, Arif Mahmood, Muhammad Haris Khan.<br />
  "Depth Attention for Robust RGB Tracking." ACCV (2024).
  [[paper](https://arxiv.org/abs/2410.20395)] 
  [[code](https://github.com/LiuYuML/Depth-Attention)]

- **DepthRefiner:** Lai, Simiao and Wang, Dong and Lu, Huchuan.<br />
  "DepthRefiner: Adapting RGB Trackers to RGBD Scenes via Depth-Fused Refinement." ICME (2024).
  [[paper](https://ieeexplore.ieee.org/document/10687717)] 

- **TABBTrack:** Ge Ying and Dawei Zhang and Zhou Ou and Xiao Wang and Zhonglong Zheng.<br />
  "Temporal adaptive bidirectional bridging for RGB-D tracking." PR (2024).
  [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320324008045)] 

- **AMATrack:** Ye, Ping and Xiao, Gang and Liu, Jun.<br />
  "AMATrack: A Unified Network With Asymmetric Multimodal Mixed Attention for RGBD Tracking." IEEE TIM (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10623547)] 

- **SSLTrack:** Xue-Feng Zhu, Tianyang Xu, Sara Atito, Muhammad Awais,
Xiao-Jun Wu, Zhenhua Feng, Josef Kittler.<br />
  "Self-supervised learning for RGB-D object tracking." PR (2024).
  [[paper](https://www.sciencedirect.com/science/article/pii/S0031320324002942)] 

- **VADT:** Zhang, Guangtong and Liang, Qihua and Mo, Zhiyi and Li, Ning and Zhong, Bineng.<br />
  "Visual Adapt for RGBD Tracking." ICASSP (2024).
  [[paper](https://ieeexplore.ieee.org/document/10447728)] 

- **FECD:** Xue-Feng Zhu, Tianyang Xu, Xiao-Jun Wu, Josef Kittler.<br />
  "Feature enhancement and coarse-to-fine detection for RGB-D tracking." PRL (2024).
  [[paper](https://www.sciencedirect.com/science/article/pii/S0167865524000412)] 

- **CDAAT:** Xue-Feng Zhu, Tianyang Xu, Xiao-Jun Wu, Zhenhua Feng, Josef Kittler.<br />
  "Adaptive Colour-Depth Aware Attention for RGB-D Object Tracking." SPL (2024).
  [[paper](https://ieeexplore.ieee.org/document/10472092/)] 
  [[code](https://github.com/xuefeng-zhu5/CDAAT)]


#### 2023
- **SPT:** Xue-Feng Zhu, Tianyang Xu, Zhangyong Tang, Zucheng Wu, Haodong Liu, Xiao Yang, Xiao-Jun Wu, Josef Kittler.<br />
  "RGBD1K: A Large-scale Dataset and Benchmark for RGB-D Object Tracking." AAAI (2023).
  [[paper](https://arxiv.org/pdf/2208.09787.pdf)] 
  [[code](https://github.com/xuefeng-zhu5/RGBD1K)]

- **EMT:** Yang, Jinyu and Gao, Shang and Li, Zhe and Zheng, Feng and Leonardis, Ale\v{s}.<br />
  "Resource-Effcient RGBD Aerial Tracking." CVPR (2023).
  [[paper](https://ieeexplore.ieee.org/document/10204937/)] 
  [[code](https://github.com/yjybuaa/RGBDAerialTracking)]

#### 2022
- **Track-it-in-3D:** Jinyu Yang, Zhongqun Zhang, Zhe Li, Hyung Jin Chang, Aleš Leonardis, Feng Zheng.<br />
  "Towards Generic 3D Tracking in RGBD Videos: Benchmark and Baseline." ECCV  (2022).
  [[paper](https://link.springer.com/chapter/10.1007/978-3-031-20047-2_7)] 
  [[code](https://github.com/yjybuaa/Track-it-in-3D)]

- **DMTracker:** Shang Gao, Jinyu Yang, Zhe Li, Feng Zheng, Aleš Leonardis, Jingkuan Song.<br />
  "Learning Dual-Fused Modality-Aware Representations for RGBD Tracking." ECCVW (2022).
  [[paper](https://arxiv.org/abs/2211.03055)] 


#### 2021

- **DeT:** Song Yan, Jinyu Yang, Jani Käpylä, Feng Zheng, Aleš Leonardis, Joni-Kristian Kämäräinen.<br />
  "DepthTrack: Unveiling the Power of RGBD Tracking." ICCV (2021).
  [[paper](https://arxiv.org/abs/2108.13962)] 
  [[code](https://github.com/xiaozai/DeT)]

- **TSDM:** Pengyao Zhao, Quanli Liu, Wei Wang and Qiang Guo.<br />
  "TSDM: Tracking by SiamRPN++ with a Depth-refiner and a Mask-generator." ICPR (2021).
  [[paper](https://arxiv.org/ftp/arxiv/papers/2005/2005.04063.pdf)] 
  [[code](https://github.com/lql-team/TSDM)]

- **3s-RGBD:** Feng Xiao, Qiuxia Wu, Han Huang.<br />
  "Single-scale siamese network based RGB-D object tracking with adaptive bounding boxes." Neurocomputing (2021).
  [[paper](https://www.sciencedirect.com/sdfe/reader/pii/S0925231221005439/pdf)] 
  

#### 2020
- **DAL:** Yanlin Qian, Alan Lukezic, Matej Kristan, Joni-Kristian Kämäräinen, Jiri Matas.<br />
  "DAL : A deep depth-aware long-term tracker." ICPR (2020).
  [[paper](https://arxiv.org/abs/1912.00660)] 
  [[code](https://github.com/xiaozai/DAL)]

- **RF-CFF:** Yong Wang, Xian Wei, Hao Shen, Lu Ding, Jiuqing Wan.<br />
  "Robust fusion for RGB-D tracking using CNN features." Applied Soft Computing Journal (2020).
  [[paper](https://www.sciencedirect.com/sdfe/reader/pii/S1568494620302428/pdf)] 

- **SiamOC:** Wenli Zhang, Kun Yang, Yitao Xin, Rui Meng.<br />
  "An Occlusion-Aware RGB-D Visual Object Tracking Method Based on Siamese Network.." ICSP (2020).
  [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9320907)] 

- **WCO:** Weichun Liu, Xiaoan Tang, Chengling Zhao.<br />
  "Robust RGBD Tracking via Weighted Convlution Operators." Sensors (2020).
  [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8950173/)] 

#### 2019
- **OTR:** Ugur Kart, Alan Lukezic, Matej Kristan, Joni-Kristian Kamarainen, Jiri Matas.<br />
  "Object Tracking by Reconstruction with View-Specific Discriminative Correlation Filters." CVPR (2019).
  [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kart_Object_Tracking_by_Reconstruction_With_View-Specific_Discriminative_Correlation_Filters_CVPR_2019_paper.pdf)] 
  [[code](https://github.com/ugurkart/OTR)]

- **H-FCN:** Ming-xin Jiang, Chao Deng, Jing-song Shan, Yuan-yuan Wang, Yin-jie Jia, Xing Sun.<br />
  "Hierarchical multi-modal fusion FCN with attention model for RGB-D tracking." Information Fusion (2019).
  [[paper](https://www.sciencedirect.com/sdfe/reader/pii/S1566253517306784/pdf)] 

- Kuai, Yangliu and Wen, Gongjian and Li, Dongdong and Xiao, Jingjing.<br />
  "Target-Aware Correlation Filter Tracking in RGBD Videos." IEEE Sensors Journal (2019).
  [[paper](https://ieeexplore.ieee.org/abstract/document/8752050)] 

- **RGBD-OD:** Yujun Xie, Yao Lu, Shuang Gu.<br />
  "RGB-D Object Tracking with Occlusion Detection." CIS (2019).
  [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9023755)] 

- **3DMS:** Alexander Gutev, Carl James Debono.<br />
  "Exploiting Depth Information to Increase Object Tracking Robustness." ICST (2019).
  [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8861628/)] 

- **CA3DMS:** Ye Liu, Xiao-Yuan Jing, Jianhui Nie, Hao Gao, Jun Liu, Guo-Ping Jiang.<br />
  "Context-Aware Three-Dimensional Mean-Shift With Occlusion Handling for Robust Object Tracking in RGB-D Videos." TMM (2019).
  [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8425768)] 
  [[code](https://github.com/yeliu2013/ca3dms-toh)]

- **Depth-CCF:** Guanqun Li, Lei Huang, Peichang Zhang, Qiang Li, YongKai Huo.<br />
  "Depth Information Aided Constrained correlation Filter for Visual Tracking." GSKI  (2019).
  [[paper](https://iopscience.iop.org/article/10.1088/1755-1315/234/1/012005)] 


#### 2018
- **STC:** Jingjing Xiao, Rustam Stolkin, Yuqing Gao, Aleš Leonardis.<br />
  "Robust Fusion of Color and Depth Data for RGB-D Target Tracking Using Adaptive Range-Invariant Depth Models and Spatio-Temporal Consistency Constraints." TC (2018).
  [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8026575)] 
  [[code](https://github.com/shine636363/RGBDtracker)]

- Kart, Uğur and Kämäräinen, Joni-Kristian and Matas, Jiří.<br />
  "How to Make an RGBD Tracker ?." ECCVW (2018).
  [[paper](https://link.springer.com/chapter/10.1007/978-3-030-11009-3_8)] 
  [[code](https://github.com/ugurkart/rgbdconverter)]

- Jiaxu Leng, Ying Liu.<br />
  "Real-Time RGB-D Visual Tracking With Scale Estimation and Occlusion Handling." IEEE Access (2018).
  [[paper](https://ieeexplore.ieee.org/document/8353501)] 

- **DM-DCF:** Uğur Kart, Joni-Kristian Kämäräinen, Jiří Matas, Lixin Fan, Francesco Cricri.<br />
  "Depth Masked Discriminative Correlation Filter." ICPR (2018).
  [[paper](https://arxiv.org/pdf/1802.09227.pdf)] 

- **OACPF:** Yayu Zhai, Ping Song, Zonglei Mou, Xiaoxiao Chen, Xiongjun Liu.<br />
  "Occlusion-Aware Correlation Particle FilterTarget Tracking Based on RGBD Data." Access (2018).
  [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8463446)] 

- **RT-KCF:** Han Zhang, Meng Cai, Jianxun Li.<br />
  "A Real-time RGB-D tracker based on KCF." CCDC (2018).
  [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8407972)] 

#### 2017
- **ODIOT:** Wei-Long Zheng, Shan-Chun Shen, Bao-Liang Lu.<br />
  "Online Depth Image-Based Object Tracking with Sparse Representation and Object Detection." Neural Process Letters (2017).
  [[paper](https://link.springer.com/content/pdf/10.1007/s11063-016-9509-y.pdf)] 

- **ROTSL:** Zi-ang Ma, Zhi-yu Xiang.<br />
  "Robust Object Tracking with RGBD-based Sparse Learning." ITEE  (2017).
  [[paper](https://link.springer.com/article/10.1631/FITEE.1601338)] 

#### 2016

- **DLS:** Ning An, Xiao-Guang Zhao, Zeng-Guang Hou.<br />
  "Online RGB-D Tracking via Detection-Learning-Segmentation." ICPR  (2016).
  [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7899805)] 

- **DS-KCF_shape:** Sion Hannuna, Massimo Camplani, Jake Hall, Majid Mirmehdi, Dima Damen, Tilo Burghardt, Adeline Paiement, Lili Tao.<br />
  "DS-KCF: A Real-time Tracker for RGB-D Data." RTIP (2016).
  [[paper](https://link.springer.com/content/pdf/10.1007/s11554-016-0654-3.pdf)] 
  [[code](https://github.com/mcamplan/DSKCF_JRTIP2016)]

- **3D-T:** Adel Bibi, Tianzhu Zhang, Bernard Ghanem.<br />
  "3D Part-Based Sparse Tracker with Automatic Synchronization and Registration." CVPR  (2016).
  [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bibi_3D_Part-Based_Sparse_CVPR_2016_paper.pdf)] 
  [[code](https://github.com/adelbibi/3D-Part-Based-Sparse-Tracker-with-Automatic-Synchronization-and-Registration)]

- **OAPF:** Kourosh Meshgia, Shin-ichi Maedaa, Shigeyuki Obaa, Henrik Skibbea, Yu-zhe Lia, Shin Ishii.<br />
  "Occlusion Aware Particle Filter Tracker to Handle Complex and Persistent Occlusions." CVIU  (2016).
  [[paper](http://ishiilab.jp/member/meshgi-k/files/ai/prl14/OAPF.pdf)] 

#### 2015
- **CDG:** Huizhang Shi, Changxin Gao, Nong Sang.<br />
  "Using Consistency of Depth Gradient to Improve Visual Tracking in RGB-D sequences." CAC (2015).
  [[paper](https://ieeexplore.ieee.org/document/7382555)] 

- **DS-KCF:** Massimo Camplani, Sion Hannuna, Majid Mirmehdi, Dima Damen, Adeline Paiement, Lili Tao, Tilo Burghardt.<br />
  "Real-time RGB-D Tracking with Depth Scaling Kernelised Correlation Filters and Occlusion Handling." BMVC (2015).
  [[paper](https://core.ac.uk/reader/78861956)] 
  [[code](https://github.com/mcamplan/DSKCF_BMVC2015)]

- **DOHR:** Ping Ding, Yan Song.<br />
  "Robust Object Tracking Using Color and Depth Images with a Depth Based Occlusion Handling and Recovery." FSKD (2015).
  [[paper](https://ieeexplore.ieee.org/document/7382068)] 

- **ISOD:** Yan Chen, Yingju Shen, Xin Liu, Bineng Zhong.<br />
  "3D Object Tracking via Image Sets and Depth-Based Occlusion Detection." SP  (2015).
  [[paper](https://www.sciencedirect.com/science/article/pii/S0165168414004204)] 

- **OL3DC:** Bineng Zhong, Yingju Shen, Yan Chen, Weibo Xie, Zhen Cui, Hongbo Zhang, Duansheng Chen ,Tian Wang, Xin Liu, Shujuan Peng, Jin Gou, Jixiang Du, Jing Wang, Wenming Zheng.<br />
  "Online Learning 3D Context for Robust Visual Tracking." Neurocomputing  (2015).
  [[paper](https://www.sciencedirect.com/science/article/pii/S0925231214013757)] 


#### 2014
- **MCBT:** Qi Wang, Jianwu Fang, Yuan Yuan. Multi-Cue Based Tracking.<br />
  "Multi-Cue Based Tracking." Neurocomputing  (2014).
  [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.700.8771&rep=rep1&type=pdf)] 

#### 2013
- **PT:** Shuran Song, Jianxiong Xiao.<br />
  "Tracking Revisited using RGBD Camera: Unified Benchmark and Baselines." ICCV (2013).
  [[paper](https://vision.princeton.edu/projects/2013/tracking/paper.pdf)] 
  [[code](https://tracking.cs.princeton.edu/index.html)]

#### 2012

- Matteo Munaro, Filippo Basso and Emanuele Menegatti
.<br />
  "Tracking people within groups with RGB-D data." IROS (2012).
  [[paper](https://ieeexplore.ieee.org/abstract/document/6385772/)] 

- **AMCT:** Germán Martín García, Dominik Alexander Klein, Jörg Stückler, Simone Frintrop, Armin B. Cremers.<br />
  "Adaptive Multi-cue 3D Tracking of Arbitrary Objects." JDOS (2012).
  [[paper](https://link.springer.com/chapter/10.1007/978-3-642-32717-9_36)] 

## RGBT Tracking
### Datasets
| Dataset | Pub. & Date  | WebSite | Introduction |
|:-----:|:-----:|:-----:|:-----:|
|  [GTOT](https://ieeexplore.ieee.org/abstract/document/7577747)   |   TIP-2016   |  [GTOT](https://pan.baidu.com/s/1QNidEo-HepRaS6OIZr7-Cw)  |  50 video pairs, 1.5W frames |  
|  [RGBT210](https://dl.acm.org/doi/10.1145/3123266.3123289)   |   ACM MM-2017   |  [RGBT210](https://drive.google.com/file/d/0B3i2rdXLNbdUTkhsLVRwcTBTMlU/view?resourcekey=0-vytg_w3hqlQfLhoiS2J8Dg)  |  210 video pairs  |  
|  [RGBT234](https://arxiv.org/abs/1805.08982)   |   PR-2018   |  [RGBT234](https://sites.google.com/view/ahutracking001/)  |   234 video pairs, the extension of RGBT210  |  
|  [LasHeR](https://arxiv.org/pdf/2104.13202.pdf)   |   TIP-2021  |  [LasHeR](https://github.com/BUGPLEASEOUT/LasHeR)  | 1224 video pairs, 730K frames  |  
|  [VTUAV](https://arxiv.org/pdf/2204.04120.pdf)   |   CVPR-2022   |  [VTUAV](https://zhang-pengyu.github.io/DUT-VTUAV/)  |  Visible-thermal UAV tracking, 500 sequences, 1.7 million high-resolution frame pairs |  
|  [MV-RGBT](https://arxiv.org/abs/2405.00168)   |   arXiv-2024   |  [MV-RGBT](https://github.com/Zhangyong-Tang/MoETrack)  | 122 video pairs, 89.9K frames   |  



### Papers

#### 2025
- **TUFNet:**  Yisong Liu, Zhao Gao, Yang Cao, Dongming Zhou.<br />
  " Two-stage Unidirectional Fusion Network for RGBT tracking." KBS (2025).
  [[paper](https://www.sciencedirect.com/science/article/pii/S0950705125000310)] 

- **MAT:** He Wang, Tianyang Xu, Zhangyong Tang, Xiao-Jun Wu, Josef Kittler.<br />
  "Multi-modal adapter for RGB-T tracking." Information Fusion (2025).
  [[paper](https://www.sciencedirect.com/science/article/pii/S1566253525000132)]
  [[code](https://github.com/ouha1998/MAT.git)]

- **BTMTrack:** Zhongxuan Zhang, Bi Zeng, Xinyu Ni, Yimin Du.<br />
  "BTMTrack: Robust RGB-T Tracking via Dual-template Bridging and Temporal-Modal Candidate Elimination." ArXiv (2025).
  [[paper](https://arxiv.org/abs/2501.03616)] 

#### 2024
- **STMT:** Sun, Dengdi and Pan, Yajie and Lu, Andong and Li, Chenglong and Luo, Bin.<br />
  "Transformer RGBT Tracking With Spatio-Temporal Multimodal Tokens." TCSVT (2024).
  [[paper](https://ieeexplore.ieee.org/document/10589660)] 
  [[code](https://github.com/yinghaidada/STMT)]


- **TGTrack:** Chen, Liang and Zhong, Bineng and Liang, Qihua and Zheng, Yaozong and Mo, Zhiyi and Song, Shuxiang.<br />
  "Top-Down Cross-Modal Guidance for Robust RGB-T Tracking." TCSVT (2024).
  [[paper](https://ieeexplore.ieee.org/document/10614652)] 

- **MCTrack:** Hu, Xiantao and Zhong, Bineng and Liang, Qihua and Zhang, Shengping and Li, Ning and Li, Xianxian.<br />
  "Toward Modalities Correlation for RGB-T Tracking." TCSVT (2024).
  [[paper](https://ieeexplore.ieee.org/document/10517645)] 


- **LSAR:** Liu, Jun and Luo, Zhongqiang and Xiong, Xingzhong.<br />
  "Online Learning Samples and Adaptive Recovery for Robust RGB-T Tracking." TCSVT (2024).
  [[paper](https://ieeexplore.ieee.org/document/10159404)] 

- **SiamTFA:** Zhang, Jianming and Qin, Yu and Fan, Shimeng and Xiao, Zhu and Zhang, Jin.<br />
  "SiamTFA: Siamese Triple-Stream Feature Aggregation Network for Efficient RGBT Tracking." TITS (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10804856)] 
  [[code](https://github.com/zjjqinyu/SiamTFA)]

- **DKDTrack:** Fanghua Hong, Mai Wen, Andong Lu, Qunjing Wang.<br />
  "DKDTrack: dual-granularity knowledge distillation for RGBT tracking." ICGIP (2024).
  [[paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13539/135392J/DKDTrack-dual-granularity-knowledge-distillation-for-RGBT-tracking/10.1117/12.3057742.short)] 

- Fanghua Hong, Jinhu Wang, Andong Lu, Qunjing Wang.<br />
  "Augmentative fusion network for robust RGBT tracking." ICGIP (2024).
  [[paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13539/135392I/Augmentative-fusion-network-for-robust-RGBT-tracking/10.1117/12.3057761.short)] 

- **CAFF:**  FENG Zihang, et al.<br />
  "A content-aware correlation filter with multi-feature fusion for RGB-T tracking." Journal of Systems Engineering and Electronics (2024).
  [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10530492)]
  
- **DDFNe:** Chenglong Li, Tao Wang, Zhaodong Ding, Yun Xiao, Jin Tang.<br />
  "Dynamic Disentangled Fusion Network for RGBT Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2412.08441)] 

- **TMTB:** Yimin Du, Bi Zeng, Qingmao Wei, Boquan Zhang & Huiting Hu.<br />
  "Transformer-Mamba-Based Trident-Branch RGB-T Tracker." PRICAI (2024).
  [[paper](https://link.springer.com/chapter/10.1007/978-981-96-0122-6_4)] 

- Shuixin Pan and Haopeng Wang and Dilong Li and Yueqiang Zhang and Bahubali Shiragapur and Xiaolin Liu and Qifeng Yu.<br />
  "A Lightweight Robust RGB-T Object Tracker Based on Jitter Factor and Associated Kalman Filter." Information Fusion (2024).
  [[paper](https://www.sciencedirect.com/science/article/pii/S1566253524006201)] 

- **SiamSCR:** Liu, Yisong and Zhou, Dongming and Cao, Jinde and Yan, Kaixiang and Geng, Lizhi.<br />
  "Specific and Collaborative Representations Siamese Network for RGBT Tracking." IEEE Sensors Journal (2024).
  [[paper](https://ieeexplore.ieee.org/document/10500316)] 

- Jianming Zhang, Jing Yang, Zikang Liu, Jin Wang.<br />
  "RGBT tracking via frequency-aware feature enhancement and unidirectional mixed attention." Neurocomputing (2024).
  [[paper](https://www.sciencedirect.com/science/article/pii/S0925231224016795)] 

- Jie Yu, Tianyang Xu, Xuefeng Zhu, Xiao-Jun Wu .<br />
  "Local Point Matching for Collaborative Image Registration and RGBT Anti-UAV Tracking." PRCV (2024).
  [[paper](https://link.springer.com/chapter/10.1007/978-981-97-8858-3_29)] 
  [[code](https://github.com/muqiu791/prcv)]

- **FHAT:** Lei Lei, Xianxian Li.<br />
  "RGB-T tracking with frequency hybrid awareness." Image and Vision Computing (2024).
  [[paper](https://www.sciencedirect.com/science/article/pii/S0262885624004359)] 

- **ACENet:** Zhengzheng Tu, Le Gu, Danying Lin, Zhicheng Zhao.<br />
  "ACENet: Adaptive Context Enhancement Network for RGB-T Video Object Detection." PRCV (2024).
  [[paper](https://link.springer.com/chapter/10.1007/978-981-97-8685-5_8)] 
  [[code](https://github.com/bscs12/ACENet)]

- **MMSTC:** Zhang, Tianlu and Jiao, Qiang and Zhang, Qiang and Han, Jungong.<br />
  "Exploring Multi-Modal Spatial–Temporal Contexts for High-Performance RGB-T Tracking." TIP (2024).
  [[paper](https://ieeexplore.ieee.org/document/10605602)] 

- **CKD:** Andong Lu, Jiacong Zhao, Chenglong Li, Yun Xiao, Bin Luo.<br />
  "Breaking Modality Gap in RGBT Tracking: Coupled Knowledge Distillation." ACM MM (2024).
  [[paper](https://arxiv.org/abs/2410.11586)] 
  [[code](https://github.com/Multi-Modality-Tracking/CKD)]

- **TBSI:** Li, Bo and Peng, Fengguang and Hui, Tianrui and Wei, Xiaoming and Wei, Xiaolin and Zhang, Lijun and Shi, Hang and Liu, Si.<br />
  "RGB-T Tracking with Template-Bridged Search Interaction and Target-Preserved Template Updating." TPAMI (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10706882)] 
  [[code](https://github.com/RyanHTR/TBSI)]

- **CFBT:** Zhirong Zeng, Xiaotao Liu, Meng Sun, Hongyu Wang, Jing Liu.<br />
  "Cross Fusion RGB-T Tracking with Bi-directional Adapter." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2408.16979)] 

- **MambaVT:** Simiao Lai, Chang Liu, Jiawen Zhu, Ben Kang, Yang Liu, Dong Wang, Huchuan Lu.<br />
  "MambaVT: Spatio-Temporal Contextual Modeling for robust RGB-T Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2408.07889)] 

- **DFM:** Andong Lu, Wanyu Wang, Chenglong Li, Jin Tang, Bin Luo.<br />
  "RGBT Tracking via All-layer Multimodal Interactions with Progressive Fusion Mamba." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2408.08827)]

- **SiamSEA:** Zihan Zhuang, Mingfeng Yin, Qi Gao, Yong Lin, Xing Hong.<br />
  "SiamSEA: Semantic-aware Enhancement and Associative-attention Dual-Modal Siamese Network for Robust RGBT Tracking." IEEE Access (2024).
  [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10634538)] 
  
- **VLCTrack:** Wang, Jiahao and Liu, Fang and Jiao, Licheng and Gao, Yingjia and Wang, Hao and Li, Shuo and Li, Lingling and Chen, Puhua and Liu, Xu.<br />
  "Visual and Language Collaborative Learning for RGBT Object Tracking." TCSVT (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10620225)] 

- **CAFormer:** Yun Xiao, Jiacong Zhao, Andong Lu, Chenglong Li, Yin Lin, Bing Yin, Cong Liu.<br />
  "Cross-modulated Attention Transformer for RGBT Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2408.02222)] 

- Li, Kai, Lihua Cai, Guangjian He, and Xun Gong.<br />
  "MATI: Multimodal Adaptive Tracking Integrator for Robust Visual Object Tracking." Sensors (2024).
  [[paper](https://scholar.google.com/scholar_url?url=https://www.mdpi.com/1424-8220/24/15/4911&hl=zh-TW&sa=X&d=14839197097615148070&ei=BXusZp-aO4WDy9YPk62U2Ag&scisig=AFWwaeabTxc27aJsTqC6-Qs7Z4zo&oi=scholaralrt&hist=r_cpud8AAAAJ:10441837062453208129:AFWwaeY_qPHejjD8SbhwqiaY6sXf&html=&pos=0&folt=rel&fols=)] 

- **PDAT:** Qiao Li, Kanlun Tan, Qiao Liu, Di Yuan, Xin Li, Yunpeng Liu.<br />
  "Progressive Domain Adaptation for Thermal Infrared Object Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2407.19430)] 

- **ReFocus:** Lai, Simiao and Liu, Chang and Wang, Dong and Lu, Huchuan.<br />
  "Refocus the Attention for Parameter-Efficient Thermal Infrared Object Tracking." TNNLS (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10601348)] 

- **MMSTC:** Zhang, Tianlu and Jiao, Qiang and Zhang, Qiang and Han, Jungong.<br />
  "Exploring Multi-modal Spatial-Temporal Contexts for High-performance RGB-T Tracking." TIP (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10605602)] 

- **MELT:** Zhangyong Tang, Tianyang Xu, Xiao-Jun Wu, and Josef Kittler.<br />
  "Multi-Level Fusion for Robust RGBT Tracking via Enhanced Thermal Representation." ACM TOMM (2024).
  [[paper](https://dl.acm.org/doi/abs/10.1145/3678176)] 
  [[code](https://github.com/Zhangyong-Tang/MELT)]

- **NLMTrack:** Miao Yan, Ping Zhang, Haofei Zhang, Ruqian Hao, Juanxiu Liu, Xiaoyang Wang, Lin Liu.<br />
  "Enhancing Thermal Infrared Tracking with Natural Language Modeling and Coordinate Sequence Generation." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2407.08265)] 
  [[code](https://github.com/ELOESZHANG/NLMTrack)]

- Yang Luo, Xiqing Guo, Hao Li.<br />
  "From Two-Stream to One-Stream: Efficient RGB-T Tracking via Mutual Prompt Learning and Knowledge Distillation." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2403.16834)] 

- Zhao, Qian, Jun Liu, Junjia Wang, and Xingzhong Xiong.<br />
  "Real-Time RGBT Target Tracking Based on Attention Mechanism." Electronics (2024).
  [[paper](https://www.mdpi.com/2079-9292/13/13/2517)] 

- **MIGTD:** Yujue Cai, Xiubao Sui, Guohua Gu, Qian Chen.<br />
  "Multi-modal interaction with token division strategy for RGB-T tracking." PR (2024).
  [[paper](https://www.sciencedirect.com/science/article/pii/S0031320324003777)] 

- **GMMT:** Zhangyong Tang, Tianyang Xu, Xuefeng Zhu, Xiao-Jun Wu, Josef Kittler.<br />
  "Generative-based Fusion Mechanism for Multi-Modal Tracking." AAAI (2024).
  [[paper](https://arxiv.org/abs/2309.01728)] 
  [[code](https://github.com/Zhangyong-Tang/GMMT)]

- **BAT:** Bing Cao, Junliang Guo, Pengfei Zhu, Qinghua Hu.<br />
  "Bi-directional Adapter for Multi-modal Tracking." AAAI (2024).
  [[paper](https://arxiv.org/abs/2312.10611)] 
  [[code](https://github.com/SparkTempest/BAT)]

- **ProFormer:** Yabin Zhu, Chenglong Li, Xiao Wang, Jin Tang, Zhixiang Huang.<br />
  "RGBT Tracking via Progressive Fusion Transformer with Dynamically Guided Learning." TCSVT (2024).
  [[paper](https://ieeexplore.ieee.org/document/10506555/)] 

- **QueryTrack:** Fan, Huijie and Yu, Zhencheng and Wang, Qiang and Fan, Baojie and Tang, Yandong.<br />
  "QueryTrack: Joint-Modality Query Fusion Network for RGBT Tracking." TIP (2024).
  [[paper](https://ieeexplore.ieee.org/document/10516307)] 

- **CAT++:** Liu, Lei and Li, Chenglong and Xiao, Yun and Ruan, Rui and Fan, Minghao.<br />
  "RGBT Tracking via Challenge-Based Appearance Disentanglement and Interaction." TIP (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10460420)] 

- **TATrack:** Hongyu Wang, Xiaotao Liu, Yifan Li, Meng Sun, Dian Yuan, Jing Liu.<br />
  "Temporal Adaptive RGBT Tracking with Modality Prompt." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2401.01244)] 

- **MArMOT:** Chenglong Li, Tianhao Zhu, Lei Liu, Xiaonan Si, Zilin Fan, Sulan Zhai.<br />
  "Cross-Modal Object Tracking: Modality-Aware Representations and A Unified Benchmark." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2111.04264)] 

- **AMNet:** Zhang, Tianlu and He, Xiaoyi and Jiao, Qiang and Zhang, Qiang and Han, Jungong.<br />
  "AMNet: Learning to Align Multi-modality for RGB-T Tracking." TCSVT (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10472533)] 

- **MCTrack:** Hu, Xiantao and Zhong, Bineng and Liang, Qihua and Zhang, Shengping and Li, Ning and Li, Xianxian.<br />
  "Towards Modalities Correlation for RGB-T Tracking." TCSVT (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10517645)] 

- **AFter:** Andong Lu, Wanyu Wang, Chenglong Li, Jin Tang, Bin Luo.<br />
  "AFter: Attention-based Fusion Router for RGBT Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2405.02717)] 
  [[code](https://github.com/Alexadlu/AFter)]

- **CSTNet:** Yunfeng Li, Bo Wang, Ye Li, Zhiwen Yu, Liang Wang.<br />
  "Transformer-based RGB-T Tracking with Channel and Spatial Feature Fusion." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2405.03177)] 
  [[code](https://github.com/LiYunfengLYF/CSTNet)]

#### 2023
- **TBSI:** Hui, Tianrui and Xun, Zizheng and Peng, Fengguang and Huang, Junshi and Wei, Xiaoming and Wei, Xiaolin and Dai, Jiao and Han, Jizhong and Liu, Si.<br />
  "Bridging Search Region Interaction with Template for RGB-T Tracking." CVPR (2023).
  [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Hui_Bridging_Search_Region_Interaction_With_Template_for_RGB-T_Tracking_CVPR_2023_paper.pdf)] 
  [[code](https://github.com/RyanHTR/TBSI)]

- **DFNet:** Jingchao Peng , Haitao Zhao , and Zhengwei Hu.<br />
  "Dynamic Fusion Network for RGBT Tracking." TITS (2023).
  [[paper](https://arxiv.org/abs/2109.07662)] 
  [[code](https://github.com/PengJingchao/DFNet)]

- **CMD:** Zhang, Tianlu and Guo, Hongyuan and Jiao, Qiang and Zhang, Qiang and Han, Jungong.<br />
  "Efficient RGB-T Tracking via Cross-Modality Distillation." CVPR (2023).
  [[paper](https://ieeexplore.ieee.org/document/10205202)] 

- **DFAT:** Zhangyong Tang, Tianyang Xu, Hui Li, Xiao-Jun Wu, XueFeng Zhu, Josef Kittler.<br />
  "Exploring fusion strategies for accurate RGBT visual object tracking." Information Fusion (2023).
  [[paper](https://arxiv.org/abs/2201.08673)] 
  [[code](https://github.com/Zhangyong-Tang/DFAT)]

- **QAT:** Lei Liu, Chenglong Li, Yun Xiao, Jin Tang.<br />
  "Quality-Aware RGBT Tracking via Supervised Reliability Learning and Weighted Residual Guidance." ACM MM  (2023).
  [[paper](https://dl.acm.org/doi/10.1145/3581783.3612341)] 

- **GuideFuse:** Zhang, Zeyang and Li, Hui and Xu, Tianyang and Wu, Xiao-Jun and Fu, Yu.<br />
  "GuideFuse: A Novel Guided Auto-Encoder Fusion Network for Infrared and Visible Images." TIM (2023).
  [[paper](https://ieeexplore.ieee.org/document/10330731)] 

- **MPLT:** Yang Luo, Xiqing Guo, Hui Feng, Lei Ao.<br />
  "RGB-T Tracking via Multi-Modal Mutual Prompt Learning." ArXiv (2023).
  [[paper](https://arxiv.org/abs/2308.16386)] 
  [[code](https://github.com/HusterYoung/MPLT)]

#### 2022
- **HMFT:** Pengyu Zhang, Jie Zhao, Dong Wang, Huchuan Lu, Xiang Ruan.<br />
  "Visible-Thermal UAV Tracking: A Large-Scale Benchmark and New Baseline." CVPR (2022).
  [[paper](https://arxiv.org/abs/2204.04120)] 
  [[code](https://github.com/zhang-pengyu/HMFT)]

- **MFGNet:** Xiao Wang, Xiujun Shu, Shiliang Zhang, Bo Jiang, Yaowei Wang, Yonghong Tian, Feng Wu.<br />
  "MFGNet: Dynamic Modality-Aware Filter Generation for RGB-T Tracking." TMM  (2022).
  [[paper](https://arxiv.org/abs/2107.10433)] 
  [[code](https://github.com/wangxiao5791509/MFG_RGBT_Tracking_PyTorch)]

- **MBAFNet:** Li, Yadong and Lai, Huicheng and Wang, Liejun and Jia, Zhenhong.<br />
  "Multibranch Adaptive Fusion Network for RGBT Tracking." IEEE Sensors Journal (2022).
  [[paper](https://ieeexplore.ieee.org/document/9721310)] 

- **AGMINet:** Mei, Jiatian and Liu, Yanyu and Wang, Changcheng and Zhou, Dongming and Nie, Rencan and Cao, Jinde.<br />
  "Asymmetric Global–Local Mutual Integration Network for RGBT Tracking." TIM (2022).
  [[paper](https://ieeexplore.ieee.org/abstract/document/9840392/)] 

- **APFNet:** Yun Xiao, Mengmeng Yang, Chenglong Li, Lei Liu, Jin Tang.<br />
  "Attribute-Based Progressive Fusion Network for RGBT Tracking." AAAI (2022).
  [[paper](https://cdn.aaai.org/ojs/20187/20187-13-24200-1-2-20220628.pdf)] 
  [[code](https://github.com/yangmengmeng1997/APFNet)]

- **DMCNet:** Lu, Andong and Qian, Cun and Li, Chenglong and Tang, Jin and Wang, Liang.<br />
  "Duality-Gated Mutual Condition Network for RGBT Tracking." TNNLS (2022).
  [[paper](https://ieeexplore.ieee.org/document/9737634)] 

- **TFNet:** Zhu, Yabin and Li, Chenglong and Tang, Jin and Luo, Bin and Wang, Liang.<br />
  "RGBT Tracking by Trident Fusion Network." TCSVT (2022).
  [[paper](https://ieeexplore.ieee.org/document/9383014)] 

- Mingzheng Feng, Jianbo Su
  .<br />
  "Learning reliable modal weight with transformer for robust RGBT tracking." KBS (2022).
  [[paper](https://www.sciencedirect.com/science/article/pii/S0950705122004579)] 


#### 2021
- **JMMAC:** Zhang, Pengyu and Zhao, Jie and Bo, Chunjuan and Wang, Dong and Lu, Huchuan and Yang, Xiaoyun.<br />
  "Jointly Modeling Motion and Appearance Cues for Robust RGB-T Tracking." TIP (2021).
  [[paper](https://ieeexplore.ieee.org/document/9364880/)] 
  [[code](https://github.com/zhang-pengyu/JMMAC)]

- **ADRNet:** Pengyu Zhang, Dong Wang, Huchuan Lu, Xiaoyun Yang.<br />
  "Learning Adaptive Attribute-Driven Representation for Real-Time RGB-T Tracking." IJCV (2021).
  [[paper](https://github.com/zhang-pengyu/ADRNet/blob/main/Zhang_IJCV2021_ADRNet.pdf)] 
  [[code](https://github.com/zhang-pengyu/ADRNet)]

- **SiamCDA:** Zhang, Tianlu and Liu, Xueru and Zhang, Qiang and Han, Jungong.<br />
  "SiamCDA: Complementarity-and distractor-aware RGB-T tracking based on Siamese network." TCSVT (2021).
  [[paper](https://ieeexplore.ieee.org/abstract/document/9399460/)] 
  [[code](https://github.com/Tianlu-Zhang/LSS-Dataset)]

- Wang, Yong and Wei, Xian and Tang, Xuan and Shen, Hao and Zhang, Huanlong.<br />
  "Adaptive Fusion CNN Features for RGBT Object Tracking." TITS (2021).
  [[paper](https://ieeexplore.ieee.org/document/9426573)] 

- **M5L:** Zhengzheng Tu, Chun Lin, Chenglong Li, Jin Tang, Bin Luo.<br />
  "M5L: Multi-Modal Multi-Margin Metric Learning for RGBT Tracking." TIP (2021).
  [[paper](https://arxiv.org/abs/2003.07650)] 

- **CBPNet:** Qin Xu, Yiming Mei, Jinpei Liu, and Chenglong Li.<br />
  "Multimodal Cross-Layer Bilinear Pooling for RGBT Tracking." TMM (2021).
  [[paper](https://ieeexplore.ieee.org/document/9340007/)] 

- **MANet++:** Andong Lu, Chenglong Li, Yuqing Yan, Jin Tang, Bin Luo.<br />
  "RGBT Tracking via Multi-Adapter Network with Hierarchical Divergence Loss." TIP (2021).
  [[paper](https://arxiv.org/abs/2011.07189)] 

- **CMR:** Li, Chenglong and Xiang, Zhiqiang and Tang, Jin and Luo, Bin and Wang, Futian.<br />
  "RGBT Tracking via Noise-Robust Cross-Modal Ranking." TNNLS (2021).
  [[paper](https://ieeexplore.ieee.org/document/9406193/)] 

- **GCMP:** Rui Yang, Xiao Wang, Chenglong Li, Jinmin Hu, Jin Tang.<br />
  "RGBT tracking via cross-modality message passing." Neurocomputing (2021).
  [[paper](https://dl.acm.org/doi/10.1016/j.neucom.2021.08.012)] 

- **HDINet:** Mei, Jiatian and Zhou, Dongming and Cao, Jinde and Nie, Rencan and Guo, Yanbu.<br />
  "HDINet: Hierarchical Dual-Sensor Interaction Network for RGBT Tracking." IEEE Sensors Journal (2021).
  [[paper](https://ieeexplore.ieee.org/abstract/document/9426927)] 

#### 2020
- **CMPP:** Chaoqun Wang, Chunyan Xu, Zhen Cui, Ling Zhou, Tong Zhang, Xiaoya Zhang, Jian Yang.<br />
  "Cross-Modal Pattern-Propagation for RGB-T Tracking."CVPR (2020).
  [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Modal_Pattern-Propagation_for_RGB-T_Tracking_CVPR_2020_paper.pdf)] 

- **CAT:** Chenglong Li, Lei Liu, Andong Lu, Qing Ji, Jin Tang.<br />
  "Challenge-Aware RGBT Tracking." ECCV (2020).
  [[paper](https://ar5iv.labs.arxiv.org/abs/2007.13143)] 

- **FANet:** Yabin Zhu, Chenglong Li, Bin Luo, Jin Tang .<br />
 "FANet: Quality-Aware Feature Aggregation Network for Robust RGB-T Tracking." TIV (2020).
  [[paper](https://arxiv.org/abs/1811.09855)] 


#### 2019
- **mfDiMP:** Lichao Zhang, Martin Danelljan, Abel Gonzalez-Garcia, Joost van de Weijer, Fahad Shahbaz Khan.<br />
  "Multi-Modal Fusion for End-to-End RGB-T Tracking." ICCVW (2019).
  [[paper](https://arxiv.org/abs/1908.11714)] 
  [[code](https://github.com/zhanglichao/end2end_rgbt_tracking)]

- **DAPNet:** Yabin Zhu, Chenglong Li, Bin Luo, Jin Tang, Xiao Wang.<br />
  "Dense Feature Aggregation and Pruning for RGBT Tracking." ACM MM (2019).
  [[paper](https://arxiv.org/abs/1907.10451)] 

- **DAFNet:** Yuan Gao, Chenglong Li, Yabin Zhu, Jin Tang, Tao He, Futian Wang.<br />
  "Deep Adaptive Fusion Network for High Performance RGBT Tracking." ICCVW (2019).
  [[paper](https://openaccess.thecvf.com/content_ICCVW_2019/html/VISDrone/Gao_Deep_Adaptive_Fusion_Network_for_High_Performance_RGBT_Tracking_ICCVW_2019_paper.html)] 
  [[code](https://github.com/mjt1312/DAFNet)]

- **MANet:** Chenglong Li, Andong Lu, Aihua Zheng, Zhengzheng Tu, Jin Tang.<br />
  "Multi-Adapter RGBT Tracking." ICCV (2019).
  [[paper](https://arxiv.org/abs/1907.07485)] 
  [[code](https://github.com/Alexadlu/MANet)]



## Miscellaneous
### Datasets
| Dataset | Pub. & Date  | WebSite | Introduction |
|:-----:|:-----:|:-----:|:-----:|
|  [WebUAV-3M](https://arxiv.org/abs/2201.07425)   |   TPAMI-2023   |  [WebUAV-3M](https://github.com/983632847/WebUAV-3M)  |  4500 videos, 3.3 million frames, UAV tracking, Vision-language-audio |  
|  [UniMod1K](https://link.springer.com/article/10.1007/s11263-024-01999-8)   |   IJCV-2024  |  [UniMod1K](https://github.com/xuefeng-zhu5/UniMod1K)  |  1050 video  pairs, 2.5 million frames, Vision-depth-language  |  


### Papers
#### 2025
- **UASTrack:** He Wang, Tianyang Xu, Zhangyong Tang, Xiao-Jun Wu, Josef Kittler.<br />
  "UASTrack: A Unified Adaptive Selection  Framework with Modality-Customization in Single  Object Tracking." ArXiv (2025).
  [[paper](https://arxiv.org/abs/2502.18220)] 
  [[code](https://github.com/wanghe/UASTrack.)]

- **LightFC-X:** Yunfeng Li, Bo Wang, Ye Li.<br />
  "LightFC-X: Lightweight Convolutional Tracker for RGB-X Tracking." ArXiv (2025).
  [[paper](https://arxiv.org/abs/2502.18143)] 
  [[code](https://github.com/LiYunfengLYF/LightFC-X)]

- **APTrack:** Xiantao Hu, Bineng Zhong, Qihua Liang, Zhiyi Mo, Liangtao Shi, Ying Tai, Jian Yang.<br />
  "Adaptive Perception for Unified Visual Multi-modal Object Tracking." ArXiv (2025).
  [[paper](https://arxiv.org/abs/2502.06583)]
  
- **SUTrack:** Xin Chen, Ben Kang, Wanting Geng, Jiawen Zhu, Yi Liu, Dong Wang, Huchuan Lu.<br />
  "SUTrack: Towards Simple and Unified Single Object Tracking." AAAI (2025).
  [[paper](https://arxiv.org/abs/2412.19138)] 
  [[code](https://github.com/chenxin-dlut/SUTrack)]

- **STTrack:** Xiantao Hu, Ying Tai, Xu Zhao, Chen Zhao, Zhenyu Zhang, Jun Li, Bineng Zhong, Jian Yang.<br />
  "Exploiting Multimodal Spatial-temporal Patterns for Video Object Tracking." AAAI (2025).
  [[paper](https://arxiv.org/abs/2412.15691)] 
  [[code](https://github.com/NJU-PCALab/STTrack)]

#### 2024
- **EMTrack:** Liu, Chang and Guan, Ziqi and Lai, Simiao and Liu, Yang and Lu, Huchuan and Wang, Dong.<br />
  "EMTrack: Efficient Multimodal Object Tracking." TCSVT (2024).
  [[paper](https://ieeexplore.ieee.org/abstract/document/10747517)] 

- **MixRGBX:** Meng Sun and Xiaotao Liu and Hongyu Wang and Jing Liu.<br />
  "MixRGBX: Universal multi-modal tracking with symmetric mixed attention." Neurocomputing (2024).
  [[paper](https://www.sciencedirect.com/science/article/pii/S0925231224010452#sec4)] 

- **XTrack:** Yuedong Tan, Zongwei Wu, Yuqian Fu, Zhuyun Zhou, Guolei Sun, Chao Ma, Danda Pani Paudel, Luc Van Gool, Radu Timofte.<br />
  "Towards a Generalist and Blind RGB-X Tracker." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2405.17773)] 
  [[code](https://github.com/supertyd/XTrack)]

- **OneTracker:** Lingyi Hong, Shilin Yan, Renrui Zhang, Wanyun Li, Xinyu Zhou, Pinxue Guo, Kaixun Jiang, Yiting Chen, Jinglun Li, Zhaoyu Chen, Wenqiang Zhang.<br />
  "OneTracker: Unifying Visual Object Tracking with Foundation Models and Efficient Tuning." CVPR (2024).
  [[paper](https://arxiv.org/abs/2403.09634)] 

- **SDSTrack:** Xiaojun Hou, Jiazheng Xing, Yijie Qian, Yaowei Guo, Shuo Xin, Junhao Chen, Kai Tang, Mengmeng Wang, Zhengkai Jiang, Liang Liu, Yong Liu.<br />
  "SDSTrack: Self-Distillation Symmetric Adapter Learning for Multi-Modal Visual Object Tracking." CVPR (2024).
  [[paper](https://arxiv.org/abs/2403.16002)] 
  [[code](https://github.com/hoqolo/SDSTrack)]

- **Un-Track:** Zongwei Wu, Jilai Zheng, Xiangxuan Ren, Florin-Alexandru Vasluianu, Chao Ma, Danda Pani Paudel, Luc Van Gool, Radu Timofte.<br />
  "Single-Model and Any-Modality for Video Object Tracking." CVPR (2024).
  [[paper](https://arxiv.org/abs/2311.15851)] 
  [[code](https://github.com/Zongwei97/UnTrack)]

- **ELTrack:** Alansari, Mohamad and Alnuaimi, Khaled and Alansari, Sara and Werghi, Naoufel and Javed, Sajid.<br />
  "ELTrack: Correlating Events and Language for Visual Tracking." ArXiv (2024).
  [[paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4764503)] 
  [[code](https://github.com/HamadYA/ELTrack-Correlating-Events-and-Language-for-Visual-Tracking)]

- **KSTrack:** He, Yuhang and Ma, Zhiheng and Wei, Xing and Gong, Yihong.<br />
  "Knowledge Synergy Learning for Multi-Modal
  Tracking." TCSVT (2024).
  [[paper](https://ieeexplore.ieee.org/document/10388341)] 

- **SeqTrackv2:** Xin Chen, Ben Kang, Jiawen Zhu, Dong Wang, Houwen Peng, Huchuan Lu.<br />
  "Unified Sequence-to-Sequence Learning for Single- and Multi-Modal Visual Object Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2304.14394)] 
  [[code](https://github.com/chenxin-dlut/SeqTrackv2)]

#### 2023
- **ViPT:** Jiawen Zhu, Simiao Lai, Xin Chen, Dong Wang, Huchuan Lu.<br />
  "Visual Prompt Multi-Modal Tracking." CVPR (2023).
  [[paper](https://arxiv.org/abs/2303.10826)] 
  [[code](https://github.com/jiawen-zhu/ViPT)]


#### 2022

- **ProTrack:** Jinyu Yang, Zhe Li, Feng Zheng, Aleš Leonardis, Jingkuan Song.<br />
  "Prompting for Multi-Modal Tracking." ACM MM (2022).
  [[paper](https://arxiv.org/abs/2207.14571)] 


## Others
#### 2024
- **GSOT3D:** Yifan Jiao, Yunhao Li, Junhua Ding, Qing Yang, Song Fu, Heng Fan, Libo Zhang.<br />
  "GSOT3D: Towards Generic 3D Single Object Tracking in the Wild." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2412.02129)] 
  [[code](https://github.com/ailovejinx/GSOT3D)]

- **BihoT:** Hanzheng Wang, Wei Li, Xiang-Gen Xia, Qian Du.<br />
  "BihoT: A Large-Scale Dataset and Benchmark for Hyperspectral Camouflaged Object Tracking." ArXiv (2024).
  [[paper](https://arxiv.org/abs/2408.12232)]
  
- **SCANet:** Yunfeng Li, Bo Wang, Jiuran Sun, Xueyi Wu, Ye Li.<br />
  "RGB-Sonar Tracking Benchmark and Spatial Cross-Attention Transformer Tracker." TCSVT (2024).
  [[paper](https://arxiv.org/abs/2406.07189)] 
  [[code](https://github.com/LiYunfengLYF/RGBS50)]


## Awesome Repositories for MMOT
- [Awesome-Visual-Language-Tracking](https://github.com/Xuchen-Li/Awesome-Visual-Language-Tracking)
- [Vision-Language_Tracking_Paper_List](https://github.com/PeterBishop0/Vision-Language_Tracking_Paper_List)
- [VisEvent_SOT_Benchmark](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)
- [RGBD-tracking-review](https://github.com/memoryunreal/RGBD-tracking-review)
- [Datasets-and-benchmark-code](https://github.com/mmic-lcl/Datasets-and-benchmark-code)
- [RGBT-Tracking-Results-Datasets-and-Methods](https://github.com/Zhangyong-Tang/RGBT-Tracking-Results-Datasets-and-Methods)
- [Multimodal-Tracking-Survey](https://github.com/zhang-pengyu/Multimodal-Tracking-Survey)



## License 
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
