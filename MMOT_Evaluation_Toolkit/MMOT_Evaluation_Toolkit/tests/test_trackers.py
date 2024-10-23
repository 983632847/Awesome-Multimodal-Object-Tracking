from __future__ import absolute_import

import unittest
import os
import random

from got10k.trackers import IdentityTracker
from got10k.datasets import GOT10k


class TestTrackers(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'data'
        self.tracker = IdentityTracker()

    def tearDown(self):
        pass

    def test_identity_tracker(self):
        # setup dataset
        root_dir = os.path.join(self.data_dir, 'GOT-10k')
        dataset = GOT10k(root_dir, subset='val')
        # run experiment
        img_files, anno = random.choice(dataset)
        boxes, times = self.tracker.track(
            img_files, anno[0], visualize=True)
        self.assertEqual(boxes.shape, anno.shape)
        self.assertEqual(len(times), len(anno))


if __name__ == '__main__':
    unittest.main()
