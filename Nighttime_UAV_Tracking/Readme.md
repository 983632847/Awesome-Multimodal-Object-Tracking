# Vision-Language Night UAV Tracking: Datasets and Method

## Datasets

### Download Dataset
- To build vision-language night UAV tracking datasets, we annotate 518 language prompts for five existing datasets (i.e., UAVDark135, UAVDark70, NAT2021, NAT2021L, DarkTrack2021).
- Download the dataset through [Baidu Pan](https://pan.baidu.com/s/1ABd-OFuKRrBHKgmub1gwkw?pwd=VLUT), the extraction code is ***VLUT***.

The directory should have the following format:
```
├── Dataset (e.g., UAVDark135, UAVDark70, NAT2021, NAT2021L, DarkTrack2021)
    ├── Video-1
        ├── 00000001.jpg
        ├── imgs
            ├── 00000001.jpg
            ├── 00000002.jpg
            ├── 00000003.jpg
            ...
        ├── attributes.txt
        ├── groundtruth_rect.txt
        ├── language.txt

    ├── Video-2
    ├── Video-3
    ...
```


## MambaTrack

![image](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/blob/main/Nighttime_UAV_Tracking/imgs/MambaTrack.png)

### Download Code from [Baidu Pan](https://pan.baidu.com/s/1Ie4wLPVYGncIaBfjcZSd_Q?pwd=Mamb)

### Environment Settings 
* **Install environment using conda**
```
conda create -n mamba_fetrack python=3.10.13
conda activate mamba_fetrack
```


 * **Install the package for Vim**
```
conda install cudatoolkit==11.8 -c nvidia   
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118   
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc  #  conda install -c nvidia cuda-nvcc==11.8.89
conda install packaging  
pip install -r vim_requirements.txt  
```
* **Install the mamba-1.1.1 and casual-conv1d-1.1.3 for mamba**

Download the [mamba-1.1.1](https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl) and [source code](https://github.com/state-spaces/mamba/archive/refs/tags/v1.1.1.zip) and place it in the project path of MambaTrack. Go to source code and install the corresponding environment.
```
cd mamba-1.1.1
pip install .
```

Download the [casual-conv1d-1.1.3](https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.3/causal_conv1d-1.1.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl) and [source code](https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/v1.1.3.zip) and place it in the project path of MambaTrack.  Go to source code and install the corresponding environment.
```
cd ..
cd causal-conv1d-1.1.3
pip install .
```

If you encounter an error when installing causal_conv1d, please refer to this [URL](https://blog.csdn.net/weixin_45667052/article/details/136311600)
```
cd downloaded_packages/
pip install causal_conv1d-1.1.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 
```  

copy the "mamba_ssm" in Vim to your local "Anaconda envs" as "mamba-1.1.1" in this package has been changed


* **Install the package for tracking**
```
bash install.sh
```

* **Run the following command to set paths for this project**
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

* **After running this command, you can also modify paths by editing these two files**
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```


### Download Trained Weights for Model 
Download the trained model weights from [Baidu Pan](https://pan.baidu.com/s/1Ie4wLPVYGncIaBfjcZSd_Q?pwd=Mamb) and put it under `$/output/train-5datasets/checkpoints/train/mamba_fetrack/mamba_fetrack_felt/` for test directly.

* We release the model trained on five datasets (GOT-10k, LaSOT, COCO, TrackingNet, and VastTrack), which is a little different from the [arXiv paper](https://arxiv.org/abs/2411.15761).
* We found that training the model with more low-light videos yields better results, even without the low-light enhancement model.
* Existing low-light enhancement models (e.g., [RetinexMamba](https://github.com/YhuoyuH/RetinexMamba), [SCT](https://github.com/vision4robotics/SCT)) exhibit limited generalization capability on the Nighttime UAV Tracking datasets.
* Readers can use more robust low-light enhancement models based on our code.


### Training and Testing Script 
```
# Train
cd lib/train/
python run_training_MambaTrack.py

# Test
cd tracking/
python test.py
```


### Evaluation Toolkit 
* **The [MMOT Evaluation Toolkit](https://github.com/983632847/Awesome-Multimodal-Object-Tracking/tree/main/MMOT_Evaluation_Toolkit) now supports evaluation methods on the Nighttime UAV tracking datasets.**


### Acknowledgment 
This work was based on [OSTrack](https://github.com/botaoye/OSTrack) and [Mamba_FETrack](https://github.com/Event-AHU/Mamba_FETrack).


### :newspaper: Citation 
If you think this paper is helpful, please feel free to leave a star ⭐ and cite our paper:
```bibtex
@inproceedings{zhang2025mambatrack,
  title={Mambatrack: Exploiting dual-enhancement for night uav tracking},
  author={Zhang, Chunhui and Liu, Li and Wen, Hao and Zhou, Xi and Wang, Yanfeng},
  booktitle={ICASSP},
  pages={1--5},
  year={2025}
}
```






