# MMOT Evaluation Toolkit

> UPDATE:<br>
> All common tracking datasets (WebUOT-1M, WebUAV-3M, GOT-10k, OTB, VOT, UAV, TColor, DTB, NfS, LaSOT and TrackingNet) are supported.<br>

## How to Evaluate Performance?

### WebUAV-3M
For Overall, Attribute, Accuracy and UTUSC Protocol evaluations in OPE using Pre, nPre, AUC, cAUC and mAcc metrics:

```Python
# Step1. Run experiments on dataset

# Step2. Put the results in WebUAV-3M_Evaluation_Toolkit/results/Baseline_Results

# Step3. Report tracking performance

python WebUAV-3M_Overall_Evaluation.py

python WebUAV-3M_Attribute_Evaluation.py

python WebUAV-3M_Accuracy_Evaluation.py

python WebUAV-3M_UTUSC_Protocol.py
```

### WebUOT-1M
For Overall, Attribute, and Accuracy Protocol evaluations in OPE using Pre, nPre, AUC, cAUC and mAcc metrics:

```Python
# Step1. Run experiments on dataset

# Step2. Put the results in WebUAV-3M_Evaluation_Toolkit/results/Baseline_Results

# Step3. Report tracking performance

python WebUOT-1M_Overall_Evaluation.py

python WebUOT-1M_Attribute_Evaluation.py

python WebUOT-1M_Accuracy_Evaluation.py
```

### Acknowledgments
The MMOT evaluation toolkit is based on the great [[GOT-10k toolkit](https://github.com/got-10k/toolkit)] and [[WebUAV-3M toolkit](https://github.com/983632847/WebUAV-3M)]
