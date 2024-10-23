# MMOT Evaluation Toolkit

> UPDATE:<br>
> All common tracking datasets (GOT-10k, OTB, VOT, UAV, TColor, DTB, NfS, LaSOT and TrackingNet) are supported.<br>
> Support VOT2019 (ST/LT/RGBD/RGBT) downloading.<br>
> Fix the randomness in ImageNet-VID ([issue #13](https://github.com/got-10k/toolkit/issues/13)).

_Run experimenets over common tracking benchmarks (code from [siamfc](https://github.com/got-10k/siamfc/blob/master/test.py)):_

<img src="resources/sample_batch_run.jpg" width = "600" alt="sample_batch_run" align=center />

This repository contains the official python toolkit for running experiments and evaluate performance on [GOT-10k](http://got-10k.aitestunion.com/) benchmark. The code is written in pure python and is compile-free. Although we support both python2 and python3, we recommend python3 for better performance.

For convenience, the toolkit also provides unofficial implementation of dataset interfaces and tracking pipelines for [OTB (2013/2015)](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html), [VOT (2013~2018)](http://votchallenge.net), [DTB70](https://github.com/flyers/drone-tracking), [TColor128](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html), [NfS (30/240 fps)](http://ci2cv.net/nfs/index.html), [UAV (123/20L)](https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx), [LaSOT](https://cis.temple.edu/lasot/) and [TrackingNet](https://tracking-net.org/) benchmarks. It also offers interfaces for [ILSVRC VID](https://image-net.org/challenges/LSVRC/2015/#vid) and [YouTube-BoundingBox](https://research.google.com/youtube-bb/) (comming soon!) datasets.

[GOT-10k](http://got-10k.aitestunion.com/) is a large, high-diversity and one-shot database for training and evaluating generic purposed visual trackers. If you use the GOT-10k database or toolkits for a research publication, please consider citing:

```Bibtex
@ARTICLE{8922619,
  author={Huang, Lianghua and Zhao, Xin and Huang, Kaiqi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild}, 
  year={2021},
  volume={43},
  number={5},
  pages={1562-1577},
  doi={10.1109/TPAMI.2019.2957464}}
```

&emsp;\[[Project](http://got-10k.aitestunion.com/)\]\[[PDF](https://arxiv.org/abs/1810.11981)\]\[[Bibtex](http://got-10k.aitestunion.com/bibtex)\]

## Table of Contents

* [Installation](#installation)
* [Quick Start: A Concise Example](#quick-start-a-concise-example)
* [Quick Start: Jupyter Notebook for Off-the-Shelf Usage](#quick-start-jupyter-notebook-for-off-the-shelf-usage)
* [How to Define a Tracker?](#how-to-define-a-tracker)
* [How to Run Experiments on GOT-10k?](#how-to-run-experiments-on-got-10k)
* [How to Evaluate Performance?](#how-to-evaluate-performance)
* [How to Plot Success Curves?](#how-to-plot-success-curves)
* [How to Loop Over GOT-10k Dataset?](#how-to-loop-over-got-10k-dataset)
* [Issues](#issues)
* [Contributors](#contributors)

### Installation

Install the toolkit using `pip` (recommended):

```bash
pip install --upgrade got10k
```

Stay up-to-date:

```bash
pip install --upgrade git+https://github.com/got-10k/toolkit.git@master
```

Or, alternatively, clone the repository and install dependencies:

```
git clone https://github.com/got-10k/toolkit.git
cd toolkit
pip install -r requirements.txt
```

Then directly copy the `got10k` folder to your workspace to use it.

### Quick Start: A Concise Example

Here is a simple example on how to use the toolkit to define a tracker, run experiments on GOT-10k and evaluate performance.

```Python
from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k

class IdentityTracker(Tracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(name='IdentityTracker')
    
    def init(self, image, box):
        self.box = box

    def update(self, image):
        return self.box

if __name__ == '__main__':
    # setup tracker
    tracker = IdentityTracker()

    # run experiments on GOT-10k (validation subset)
    experiment = ExperimentGOT10k('data/GOT-10k', subset='val')
    experiment.run(tracker, visualize=True)

    # report performance
    experiment.report([tracker.name])
```

To run experiments on [OTB](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html), [VOT](http://votchallenge.net) or other benchmarks, simply change `ExperimentGOT10k`, e.g., to `ExperimentOTB` or `ExperimentVOT`, and `root_dir` to their corresponding paths for this purpose.

### Quick Start: Jupyter Notebook for Off-the-Shelf Usage

Open [quick_examples.ipynb](https://github.com/got-10k/toolkit/tree/master/examples/quick_examples.ipynb) in [Jupyter Notebook](http://jupyter.org/) to see more examples on toolkit usage.

### How to Define a Tracker?

To define a tracker using the toolkit, simply inherit and override `init` and `update` methods from the `Tracker` class. Here is a simple example:

```Python
from got10k.trackers import Tracker

class IdentityTracker(Tracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(
            name='IdentityTracker',  # tracker name
            is_deterministic=True    # stochastic (False) or deterministic (True)
        )
    
    def init(self, image, box):
        self.box = box

    def update(self, image):
        return self.box
```

### How to Run Experiments on GOT-10k?

Instantiate an `ExperimentGOT10k` object, and leave all experiment pipelines to its `run` method:

```Python
from got10k.experiments import ExperimentGOT10k

# ... tracker definition ...

# instantiate a tracker
tracker = IdentityTracker()

# setup experiment (validation subset)
experiment = ExperimentGOT10k(
    root_dir='data/GOT-10k',    # GOT-10k's root directory
    subset='val',               # 'train' | 'val' | 'test'
    result_dir='results',       # where to store tracking results
    report_dir='reports'        # where to store evaluation reports
)
experiment.run(tracker, visualize=True)
```

The tracking results will be stored in `result_dir`.

### How to Evaluate Performance?

Use the `report` method of `ExperimentGOT10k` for this purpose:

```Python
# ... run experiments on GOT-10k ...

# report tracking performance
experiment.report([tracker.name])
```

When evaluated on the __validation subset__, the scores and curves will be directly generated in `report_dir`.

However, when evaluated on the __test subset__, since all groundtruths are withholded, you will have to submit your results to the [evaluation server](http://got-10k.aitestunion.com/submit_instructions) for evaluation. The `report` function will generate a `.zip` file which can be directly uploaded for submission. For more instructions, see [submission instruction](http://got-10k.aitestunion.com/submit_instructions).

See public evaluation results on [GOT-10k's leaderboard](http://got-10k.aitestunion.com/leaderboard).

## How to Plot Success Curves?

Assume that a list of all performance files (JSON files) are stored in `report_files`, here is an example showing how to plot success curves:

```Python
from got10k.experiments import ExperimentGOT10k

report_files = ['reports/GOT-10k/performance_25_entries.json']
tracker_names = ['SiamFCv2', 'GOTURN', 'CCOT', 'MDNet']

# setup experiment and plot curves
experiment = ExperimentGOT10k('data/GOT-10k', subset='test')
experiment.plot_curves(report_files, tracker_names)
```

The report file of 25 baseline entries can be downloaded from the [Downloads page](http://got-10k.aitestunion.com/downloads). You can also download single report file for each entry from the [Leaderboard page](http://got-10k.aitestunion.com/leaderboard).

### How to Loop Over GOT-10k Dataset?

The `got10k.datasets.GOT10k` provides an iterable and indexable interface for GOT-10k's sequences. Here is an example:

```Python
from PIL import Image
from got10k.datasets import GOT10k
from got10k.utils.viz import show_frame

dataset = GOT10k(root_dir='data/GOT-10k', subset='train')

# indexing
img_file, anno = dataset[10]

# for-loop
for s, (img_files, anno) in enumerate(dataset):
    seq_name = dataset.seq_names[s]
    print('Sequence:', seq_name)

    # show all frames
    for f, img_file in enumerate(img_files):
        image = Image.open(img_file)
        show_frame(image, anno[f, :])
```

To loop over `OTB` or `VOT` datasets, simply change `GOT10k` to `OTB` or `VOT` for this purpose.

### Issues

Please report any problems or suggessions in the [Issues](https://github.com/got-10k/toolkit/issues) page.

### Contributors

- [Lianghua Huang](https://github.com/huanglianghua)
