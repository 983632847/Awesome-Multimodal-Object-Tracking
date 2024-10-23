from __future__ import absolute_import

import unittest
import os

from got10k.trackers import IdentityTracker
from got10k.experiments import ExperimentGOT10k, ExperimentOTB, \
    ExperimentVOT, ExperimentDTB70, ExperimentTColor128, \
    ExperimentUAV123, ExperimentNfS


class TestExperiments(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'data'
        self.tracker = IdentityTracker()

    def tearDown(self):
        pass

    def test_got10k(self):
        root_dir = os.path.join(self.data_dir, 'GOT-10k')
        # run experiment
        experiment = ExperimentGOT10k(root_dir)
        experiment.run(self.tracker, visualize=False)
        # report performance
        experiment.report([self.tracker.name])

    def test_otb(self):
        root_dir = os.path.join(self.data_dir, 'OTB')
        # run experiment
        experiment = ExperimentOTB(root_dir)
        experiment.run(self.tracker, visualize=False)
        # report performance
        experiment.report([self.tracker.name])

    def test_vot(self):
        root_dir = os.path.join(self.data_dir, 'vot2018')
        # run experiment
        experiment = ExperimentVOT(root_dir)
        experiment.run(self.tracker, visualize=False)
        # report performance
        experiment.report([self.tracker.name])

    def test_dtb70(self):
        root_dir = os.path.join(self.data_dir, 'DTB70')
        # run experiment
        experiment = ExperimentDTB70(root_dir)
        experiment.run(self.tracker, visualize=False)
        # report performance
        experiment.report([self.tracker.name])

    def test_uav123(self):
        root_dir = os.path.join(self.data_dir, 'UAV123')
        for version in ['UAV123', 'UAV20L']:
            # run experiment
            experiment = ExperimentUAV123(root_dir, version)
            experiment.run(self.tracker, visualize=False)
            # report performance
            experiment.report([self.tracker.name])

    def test_nfs(self):
        root_dir = os.path.join(self.data_dir, 'nfs')
        for fps in [30, 240]:
            # run experiment
            experiment = ExperimentNfS(root_dir, fps)
            experiment.run(self.tracker, visualize=False)
            # report performance
            experiment.report([self.tracker.name])

    def test_tcolor128(self):
        root_dir = os.path.join(self.data_dir, 'Temple-color-128')
        # run experiment
        experiment = ExperimentTColor128(root_dir)
        experiment.run(self.tracker, visualize=False)
        # report performance
        experiment.report([self.tracker.name])


if __name__ == '__main__':
    unittest.main()
