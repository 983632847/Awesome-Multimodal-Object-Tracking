from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import six


class MATrack(object):
    """`WebUAV-3M <https://github.com/flyers/drone-tracking>`_ Dataset.

    Publication:
        ``WebUAV-3M: A Benchmark for Unveiling the Power of Million-Scale Deep UAV Tracking``,
        Chunhui Zhang, Guanjie Huang, Li Liu, Shan Huang, Yinan Yang, Xiang Wan,
        Shiming Ge, Dacheng Tao. arXiv 2022.

    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
    """

    def __init__(self, root_dir, return_meta=True):
        super(MATrack, self).__init__()
        self.root_dir = root_dir
        self._check_integrity(root_dir)

        self.anno_files = sorted(glob.glob(
            os.path.join(root_dir, '*/groundtruth_rect.txt')))
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]

        self.return_meta = return_meta

    def __getitem__(self, index):
        r"""
        Args:
            index (integer or string): Index or name of a sequence.

        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], 'imgs/*.jpg')))
        anno = np.loadtxt(self.anno_files[index], delimiter=',')
        ##########################################################
        # assert len(img_files) == len(anno)
        ##########################################################
        assert anno.shape[1] == 4

        if self.return_meta:
            meta = self._fetch_meta(self.seq_dirs[index])
            return img_files, anno, meta
        else:
            return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self, root_dir):
        seq_names = os.listdir(root_dir)
        seq_names = [n for n in seq_names if not n[0] == '.']

        if os.path.isdir(root_dir) and len(seq_names) > 0:
            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted.')

    def _fetch_meta(self, seq_dir):
        # seq_dir = os.path.dirname(seq_dir)
        meta = {}

        # attributes
        try:
            att_file = os.path.join(seq_dir, 'attributes.txt')
            meta['att'] = np.loadtxt(att_file, delimiter=',')
        except:
            # pass
            print("No attributes.txt")

        # # scenarios
        # try:
        #     att_file = os.path.join(seq_dir, 'scenario.txt')
        #     meta['sce'] = np.loadtxt(att_file, delimiter=',')
        # except:
        #     # pass
        #     print("No scenario.txt")

        # nlp
        try:
            nlp_file = os.path.join(seq_dir, 'language.txt')
            with open(nlp_file, 'r') as f:
                meta['nlp'] = f.read().strip()
        except:
            pass
            # print("No language.txt")
        return meta
