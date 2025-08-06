# MMOT Evaluation Toolkit

> UPDATE:<br>
> All common tracking datasets (WebUAV-3M, WebUOT-1M, UW-COT220, GOT-10k, OTB, VOT, UAV, TColor, DTB, NfS, LaSOT and TrackingNet) are supported.<br>



## Download
- You can download the MMOT evaluation toolkit from this repository, or alternatively via [Google Drive](https://drive.google.com/drive/folders/1eKPUZV5vaKwcF0gZFIkOITpEznTd7zv5?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1JygwsLTh1HbUGCdLb4LoBQ?pwd=MMOT), the extraction code is ***MMOT*** (Last updated on October 2024).
- Download annotations through [Google Drive](https://drive.google.com/drive/folders/1eKPUZV5vaKwcF0gZFIkOITpEznTd7zv5?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1-89I98ngKmRVVZjxqxH6vg?pwd=idmc).
- Download tracking results through [Google Drive](https://drive.google.com/drive/folders/1eKPUZV5vaKwcF0gZFIkOITpEznTd7zv5?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1T-VUhdW4u9Lxin78_Vo-ig?pwd=f7ia).


## How to Evaluate Performance?

### WebUAV-3M
For Overall, Attribute, Accuracy and UTUSC Protocol evaluations in OPE using Pre, nPre, AUC, cAUC and mAcc metrics:

```Python
# Step1. Run experiments on dataset

# Step2. Put the results in MMOT_Evaluation_Toolkit/results/Baseline_Results/WebUAV-3M-Test

# Step3. Report tracking performance

python WebUAV-3M_Overall_Evaluation.py

python WebUAV-3M_Attribute_Evaluation.py

python WebUAV-3M_Accuracy_Evaluation.py

python WebUAV-3M_UTUSC_Protocol.py
```

### WebUOT-1M
For Overall, Attribute, and Accuracy evaluations in OPE using Pre, nPre, AUC, cAUC and mAcc metrics:

```Python
# Step1. Run experiments on dataset

# Step2. Put the results in MMOT_Evaluation_Toolkit/results/Baseline_Results/WebUOT-1M-Test

# Step3. Report tracking performance

python WebUOT-1M_Overall_Evaluation.py

python WebUOT-1M_Attribute_Evaluation.py

python WebUOT-1M_Accuracy_Evaluation.py
```

### UW-COT220
For Overall and Accuracy evaluations in OPE using Pre, nPre, AUC, cAUC and mAcc metrics:

```Python
# Step1. Run experiments on dataset

# Step2. Put the results in MMOT_Evaluation_Toolkit/results/Baseline_Results/UW-COT220

# Step3. Report tracking performance

python UWCOT220_Overall_Evaluation.py

python UWCOT220_Accuracy_Evaluation.py
```


### Acknowledgments
The MMOT evaluation toolkit is based on the great [[GOT-10k toolkit](https://github.com/got-10k/toolkit)] and [[WebUAV-3M toolkit](https://github.com/983632847/WebUAV-3M)]
