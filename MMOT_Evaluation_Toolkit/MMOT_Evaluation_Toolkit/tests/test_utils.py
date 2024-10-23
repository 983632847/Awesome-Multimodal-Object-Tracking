from __future__ import absolute_import

import unittest
import numpy as np

from got10k.utils.metrics import rect_iou, poly_iou


class TestUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_iou(self):
        rects1 = np.random.rand(1000, 4) * 100
        rects2 = np.random.rand(1000, 4) * 100
        bound = (50, 100)
        ious1 = rect_iou(rects1, rects2, bound=bound)
        ious2 = poly_iou(rects1, rects2, bound=bound)
        self.assertTrue((ious1 - ious2).max() < 1e-14)

        polys1 = self._rect2corner(rects1)
        polys2 = self._rect2corner(rects2)
        ious3 = poly_iou(polys1, polys2, bound=bound)
        self.assertTrue((ious1 - ious3).max() < 1e-14)

    def _rect2corner(self, rects):
        x1, y1, w, h = rects.T
        x2, y2 = x1 + w, y1 + h
        corners = np.array([x1, y1, x1, y2, x2, y2, x2, y1]).T

        return corners


if __name__ == '__main__':
    unittest.main()
