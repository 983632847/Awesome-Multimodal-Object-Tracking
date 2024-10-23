from __future__ import absolute_import

import unittest
import os
import random

from got10k.datasets import GOT10k, OTB, VOT, DTB70, TColor128, \
    UAV123, NfS, LaSOT, TrackingNet, ImageNetVID


class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.expanduser('~/data')

    def tearDown(self):
        pass

    def test_got10k(self):
        root_dir = os.path.join(self.data_dir, 'GOT-10k')
        # without meta
        for subset in ['train', 'val', 'test']:
            dataset = GOT10k(root_dir, subset=subset)
            self._check_dataset(dataset)
        # with meta
        for subset in ['train', 'val', 'test']:
            dataset = GOT10k(root_dir, subset=subset, return_meta=True)
            self._check_dataset(dataset)

    def test_otb(self):
        root_dir = os.path.join(self.data_dir, 'OTB')
        dataset = OTB(root_dir)
        self._check_dataset(dataset)
    
    def test_vot(self):
        root_dir = os.path.join(self.data_dir, 'vot2018')
        # without meta
        dataset = VOT(root_dir, anno_type='rect')
        self._check_dataset(dataset)
        # with meta
        dataset = VOT(root_dir, anno_type='rect', return_meta=True)
        self._check_dataset(dataset)

    def test_dtb70(self):
        root_dir = os.path.join(self.data_dir, 'DTB70')
        dataset = DTB70(root_dir)
        self._check_dataset(dataset)
    
    def test_tcolor128(self):
        root_dir = os.path.join(self.data_dir, 'Temple-color-128')
        dataset = TColor128(root_dir)
        self._check_dataset(dataset)

    def test_uav123(self):
        root_dir = os.path.join(self.data_dir, 'UAV123')
        for version in ['UAV123', 'UAV20L']:
            dataset = UAV123(root_dir, version)
            self._check_dataset(dataset)

    def test_nfs(self):
        root_dir = os.path.join(self.data_dir, 'nfs')
        for fps in [30, 240]:
            dataset = NfS(root_dir, fps)
            self._check_dataset(dataset)

    def test_lasot(self):
        root_dir = os.path.join(self.data_dir, 'LaSOTBenchmark')
        for subset in ['train', 'test']:
            dataset = LaSOT(root_dir, subset)
            self._check_dataset(dataset)

    def test_trackingnet(self):
        root_dir = os.path.join(self.data_dir, 'TrackingNet')
        for subset in ['train', 'test']:
            dataset = TrackingNet(root_dir, subset)
            self._check_dataset(dataset)

    def test_vid(self):
        root_dir = os.path.join(self.data_dir, 'ILSVRC')
        dataset = ImageNetVID(root_dir, subset=('train', 'val'))
        self._check_dataset(dataset)
    
    def _check_dataset(self, dataset):
        n = len(dataset)
        self.assertGreater(n, 0)
        inds = random.sample(range(n), min(n, 100))
        for i in inds:
            img_files, anno = dataset[i][:2]
            self.assertEqual(len(img_files), len(anno))


if __name__ == '__main__':
    unittest.main()
