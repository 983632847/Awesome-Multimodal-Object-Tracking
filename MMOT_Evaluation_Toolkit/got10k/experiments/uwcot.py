from __future__ import absolute_import, division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import math
from PIL import Image
from scipy import signal
from ..datasets import UWCOT
from ..utils.metrics import rect_iou, center_error, center_error_norm, rect_iou_complete
from ..utils.viz import show_frame
from accuracy.accuracy import eval


class ExperimentUWCOT(object):
    r"""Experiment pipeline and evaluation toolkit for WebUAV-3M dataset.

    Args:
        root_dir (string): Root directory of WebUAV-3M dataset.
        att_name (string):
                Overall Performance is "All";
                Attribute-Based Performance are 'Low_resolution','Partial_occlusion','Full_occlusion','Out_of_view','Fast_motion',
                'Camera_motion','Viewpoint_change','Rotation','Deformation','Background_clutters',
                'Scale_variation','Aspect_ratio_variation','Illumination_variation','Motion_blur','Complexity_easy',
                'Complexity_medium','Complexity_hard','Size_small','Size_medium','Size_large',
                'Length_short','Length_medium','Length_long'.
        sce_name (string): 'Underwater'
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
        report_scenario_dir (string, optional): Directory for storing UTUSC performance
            evaluation results. Default is ``./reports_scenario``.
    """
    def __init__(self, root_dir, att_name, sce_name, return_meta=True, result_dir='results', report_dir='reports', report_scenario_dir='reports_scenario'):
        super(ExperimentUWCOT, self).__init__()
        self.return_meta = return_meta
        self.dataset = UWCOT(root_dir, self.return_meta)

        #############################################################################
        # results root
        # if sce_name == "Adversarial_examples":
        #     self.result_dir = os.path.join(result_dir, 'Baseline_Results', 'WebUAV-3M-AE')
        # else:
        #     self.result_dir = os.path.join(result_dir, 'Baseline_Results', 'WebUAV-3M')             # Test results, the first six scenarios
        self.result_dir = os.path.join(result_dir, 'Baseline_Results', 'UW-COT220')

        # report root
        self.report_dir = os.path.join(report_dir, 'UW-COT220')

        self.att_name = att_name        # if att_name == "All" for all sequences, else for each attribute
        self.sce_name = sce_name        # Scenarios

        # As nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51
        self.nbins_iou_complete = 21
        self.nbins_ce_norm = 51

        ## Attribute-Based Evaluation
        # 10+14+15+16+3=58
        self.att_dict = {
            'Low_resolution':0, 'Fast_motion':1, 'Scale_variation':2, 'Aspect_ratio_variation':3,
            'Size_Small':4, 'Size_Medium':5, 'Size_Big':6,
            'Length_Short':7, 'Length_Medium':8, 'Length_Long':9,

            'Camera_motion':10, 'Viewpoint_change':11, 'Partial_occlusion':12, 'Full_occlusion':13, 'Out_of_view':14,
            'Rotation':15, 'Deformation':16, 'Similar_distractors':17, 'Illumination_variation':18, 'Motion_blur':19,
            'Partial_target_information':20, 'Natural_object':21, 'Artificial_object':22, 'Camouflage':23,

            'Underwater_visibility_Low':24, 'Underwater_visibility_Medium':25, 'Underwater_visibility_High':26,
            'Underwater_scene_Sea':27, 'Underwater_scene_River':28, 'Underwater_scene_Pool':29, 'Underwater_scene_Water_Tank':30,
            'Underwater_scene_Fish_Tank':31, 'Underwater_scene_Basin':32, 'Underwater_scene_Bowl':33, 'Underwater_scene_Cup':34,
            'Underwater_scene_Aquaria':35, 'Underwater_scene_Pond':36, 'Underwater_scene_Puddle':37, 'Underwater_scene_Lake':38,

            'Water_color_variation_Colorless':39, 'Water_color_variation_Ash':40, 'Water_color_variation_Gray':41,
            'Water_color_variation_Green':42, 'Water_color_variation_Light_Green':43, 'Water_color_variation_Dark':44,
            'Water_color_variation_Blue_Black':45, 'Water_color_variation_Deep_Blue':46, 'Water_color_variation_Blue':47,
            'Water_color_variation_Light_Blue':48, 'Water_color_variation_Partly_Blue':49, 'Water_color_variation_GrayBlue':50,
            'Water_color_variation_Light_Yellow':51, 'Water_color_variation_Light_Brown':52, 'Water_color_variation_Cyan':53,
            'Water_color_variation_Light_Purple':54,

            'Underwater_view':55, 'Fish_eye_view':56, 'Outside_water_view':57,
        }

        self.OccN = 5   # The next K (=5) frames after the current frame

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def get_OccIndicator(self, scenario_list):
        array = np.zeros(len(scenario_list))
        idx = []
        count = []  # Count the number of occlusions
        for i in range(len(scenario_list) - 1):
            if scenario_list[i] <= scenario_list[i + 1]:
                pass
            else:
                idx.append(i)
                count.append(scenario_list[i])
        # print(count)

        for i in range(len(idx)):
            for j in range(idx[i] + 1, min(idx[i] + 1 + self.OccN, len(scenario_list))):
                if scenario_list[j] == 0:
                    array[j] = count[i]
        return array

    def run(self, tracker, visualize=False):
        print('Running tracker %s on %s...' % (
            tracker.name, type(self.dataset).__name__))

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(
                self.result_dir, tracker.name, '%s.txt' % seq_name)
            if os.path.exists(record_file):
                print('  Found results, skipping', seq_name)
                continue

            # tracking loop
            boxes, times = tracker.track(
                img_files, anno[0, :], visualize=visualize)
            assert len(boxes) == len(anno)

            # record results
            self._record(record_file, boxes, times)

    def report(self, tracker_names):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        for name in tracker_names:
            print('Evaluating', name)
            seq_num = len(self.dataset)

            # overall performance
            succ_curve = np.zeros((seq_num, self.nbins_iou))                # num of sequences * 21
            prec_curve = np.zeros((seq_num, self.nbins_ce))                 # num of sequences * 51
            succ_comp_curve = np.zeros((seq_num, self.nbins_iou_complete))  # num of sequences * 21
            prec_norm_curve = np.zeros((seq_num, self.nbins_ce_norm))       # num of sequences * 51
            speeds = np.zeros(seq_num)

            performance.update({name: {
                'overall': {},
                'seq_wise': {}
            }})

            # each address one sequence
            for s, (_, anno, meta) in enumerate(self.dataset):
                att = self.att_name       # current attribute: All, Low_resolution, ..., Length_long
                att_dict = self.att_dict  # attribute dict


                # all sequences or sequences of each attribute
                Select_FLAG = False
                if att == "All":
                    Select_FLAG = True
                else:
                    # 10+14+15+16+3=58
                    # 'Low_resolution', 'Fast_motion', 'Scale_variation', 'Aspect_ratio_variation',
                    if att_dict[att] <= 3:
                        Select_FLAG = (meta['att'][att_dict[att]] == True)

                    # 'Size_Small', 'Size_Medium', 'Size_Big',
                    elif att_dict[att] == 4:
                        Select_FLAG = (meta['att'][4] == 0)
                    elif att_dict[att] == 5:
                        Select_FLAG = (meta['att'][4] == 1)
                    elif att_dict[att] == 6:
                        Select_FLAG = (meta['att'][4] == 2)

                    # 'Length_Short', 'Length_Medium', 'Length_Long',
                    elif att_dict[att] == 7:
                        Select_FLAG = (meta['att'][5] == 0)
                    elif att_dict[att] == 8:
                        Select_FLAG = (meta['att'][5] == 1)
                    elif att_dict[att] == 9:
                        Select_FLAG = (meta['att'][5] == 2)

                    # 'Camera_motion', 'Viewpoint_change', 'Partial_occlusion', 'Full_occlusion', 'Out_of_view',
                    # 'Rotation', 'Deformation', 'Similar_distractors', 'Illumination_variation', 'Motion_blur',
                    # 'Partial_target_information',
                    elif att_dict[att]>=10 and att_dict[att]<=20:
                        Select_FLAG = (meta['att'][att_dict[att]-4] == True)  # 6-16

                    # 'Natural_object', 'Artificial_object',
                    elif att_dict[att] == 21:
                        Select_FLAG = (meta['att'][17] == 0)
                    elif att_dict[att] == 22:
                        Select_FLAG = (meta['att'][17] == 1)

                    # 'Camouflage',
                    elif att_dict[att] == 23:
                        Select_FLAG = (meta['att'][18] == True)

                    # 'Underwater_visibility_Low', 'Underwater_visibility_Medium', 'Underwater_visibility_High',
                    elif att_dict[att] == 24:
                        Select_FLAG = (meta['att'][19] == 0)
                    elif att_dict[att] == 25:
                        Select_FLAG = (meta['att'][19] == 1)
                    elif att_dict[att] == 26:
                        Select_FLAG = (meta['att'][19] == 2)

                    # 'Underwater_scene_Sea', 'Underwater_scene_River', 'Underwater_scene_Pool', 'Underwater_scene_Water_Tank',
                    # 'Underwater_scene_Fish_Tank', 'Underwater_scene_Basin', 'Underwater_scene_Bowl', 'Underwater_scene_Cup',
                    # 'Underwater_scene_Aquaria', 'Underwater_scene_Pond', 'Underwater_scene_Puddle', 'Underwater_scene_Lake',

                    elif att_dict[att] == 27:
                        Select_FLAG = (meta['att'][20] == 0)
                    elif att_dict[att] == 28:
                        Select_FLAG = (meta['att'][20] == 1)
                    elif att_dict[att] == 29:
                        Select_FLAG = (meta['att'][20] == 2)
                    elif att_dict[att] == 30:
                        Select_FLAG = (meta['att'][20] == 3)
                    elif att_dict[att] == 31:
                        Select_FLAG = (meta['att'][20] == 4)
                    elif att_dict[att] == 32:
                        Select_FLAG = (meta['att'][20] == 5)
                    elif att_dict[att] == 33:
                        Select_FLAG = (meta['att'][20] == 6)
                    elif att_dict[att] == 34:
                        Select_FLAG = (meta['att'][20] == 7)
                    elif att_dict[att] == 35:
                        Select_FLAG = (meta['att'][20] == 8)
                    elif att_dict[att] == 36:
                        Select_FLAG = (meta['att'][20] == 9)
                    elif att_dict[att] == 37:
                        Select_FLAG = (meta['att'][20] == 10)
                    elif att_dict[att] == 38:
                        Select_FLAG = (meta['att'][20] == 11)

                    # 'Water_color_variation_Colorless', 'Water_color_variation_Ash', 'Water_color_variation_Gray',
                    # 'Water_color_variation_Green', 'Water_color_variation_Light_Green', 'Water_color_variation_Dark',
                    # 'Water_color_variation_Blue_Black', 'Water_color_variation_Deep_Blue', 'Water_color_variation_Blue',
                    # 'Water_color_variation_Light_Blue', 'Water_color_variation_Partly_Blue', 'Water_color_variation_GrayBlue',
                    # 'Water_color_variation_Light_Yellow', 'Water_color_variation_Light_Brown', 'Water_color_variation_Cyan',
                    # 'Water_color_variation_Light_Purple',
                    elif att_dict[att] == 39:
                        Select_FLAG = (meta['att'][21] == 0)
                    elif att_dict[att] == 40:
                        Select_FLAG = (meta['att'][21] == 1)
                    elif att_dict[att] == 41:
                        Select_FLAG = (meta['att'][21] == 2)
                    elif att_dict[att] == 42:
                        Select_FLAG = (meta['att'][21] == 3)
                    elif att_dict[att] == 43:
                        Select_FLAG = (meta['att'][21] == 4)
                    elif att_dict[att] == 44:
                        Select_FLAG = (meta['att'][21] == 5)
                    elif att_dict[att] == 45:
                        Select_FLAG = (meta['att'][21] == 6)
                    elif att_dict[att] == 46:
                        Select_FLAG = (meta['att'][21] == 7)
                    elif att_dict[att] == 47:
                        Select_FLAG = (meta['att'][21] == 8)
                    elif att_dict[att] == 48:
                        Select_FLAG = (meta['att'][21] == 9)
                    elif att_dict[att] == 49:
                        Select_FLAG = (meta['att'][21] == 10)
                    elif att_dict[att] == 50:
                        Select_FLAG = (meta['att'][21] == 11)
                    elif att_dict[att] == 51:
                        Select_FLAG = (meta['att'][21] == 12)
                    elif att_dict[att] == 52:
                        Select_FLAG = (meta['att'][21] == 13)
                    elif att_dict[att] == 53:
                        Select_FLAG = (meta['att'][21] == 14)
                    elif att_dict[att] == 54:
                        Select_FLAG = (meta['att'][21] == 15)

                    # 'Underwater_view', 'Fish_eye_view', 'Outside_water_view',
                    elif att_dict[att] == 55:
                        Select_FLAG = (meta['att'][22] == 0)
                    elif att_dict[att] == 56:
                        Select_FLAG = (meta['att'][22] == 1)
                    elif att_dict[att] == 57:
                        Select_FLAG = (meta['att'][22] == 2)
                    else:
                        assert (att_dict[att] > 23)

                if Select_FLAG:
                    # print(s)
                    seq_name = self.dataset.seq_names[s]
                    record_file = os.path.join(self.result_dir, name, '%s.txt' % seq_name)
                    try:
                        boxes = np.loadtxt(record_file, delimiter=',')
                    except:
                        boxes = np.loadtxt(record_file, delimiter='\t')
                    boxes[0] = anno[0]

                    # #######################################################
                    # try:
                    #     boxes = np.loadtxt(record_file, delimiter=',')
                    # except:
                    #     print(record_file, "Error!")
                    #     break

                    # # not the optimal strategy
                    # if len(boxes) != len(anno):
                    #     print(record_file, "Length Error!")
                    #     # break
                    #     minLen = min(len(boxes), len(anno))
                    #     boxes = boxes[0:minLen, :]
                    #     anno = anno[0:minLen, :]
                    #######################################################
                    assert len(boxes) == len(anno)

                    # overall
                    ious, center_errors, ious_complete, center_errors_norm = self._calc_metrics(boxes, anno)   # num of frames
                    succ_curve[s], prec_curve[s], succ_comp_curve[s], prec_norm_curve[s] = self._calc_curves(ious, center_errors, ious_complete, center_errors_norm)   # num of sequences * 21, num of sequences * 51

                    # calculate average tracking speed
                    time_file = os.path.join(
                        self.result_dir, name, 'times/%s_time.txt' % seq_name)
                    if os.path.isfile(time_file):
                        times = np.loadtxt(time_file)
                        times = times[times > 0]
                        if len(times) > 0:
                            speeds[s] = np.mean(1. / times)

                    # store sequence-wise performance
                    performance[name]['seq_wise'].update({seq_name: {
                        'success_curve': succ_curve[s].tolist(),
                        'precision_curve': prec_curve[s].tolist(),
                        'success_complete_curve': succ_comp_curve[s].tolist(),
                        'precision_norm_curve': prec_norm_curve[s].tolist(),
                        'success_score': np.mean(succ_curve[s]),
                        'precision_score': prec_curve[s][20],
                        'success_rate': succ_curve[s][self.nbins_iou // 2],
                        'success_complete_score': np.mean(succ_comp_curve[s]),
                        'precision_norm_score': prec_norm_curve[s][20],
                        'success_complete_rate': succ_comp_curve[s][self.nbins_iou_complete // 2],
                        'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

            # overall
            succ_curve = np.mean(succ_curve, axis=0)     # 21
            prec_curve = np.mean(prec_curve, axis=0)     # 51
            succ_score = np.mean(succ_curve)
            prec_score = prec_curve[20]
            succ_rate = succ_curve[self.nbins_iou // 2]
            succ_comp_curve = np.mean(succ_comp_curve, axis=0)
            prec_norm_curve = np.mean(prec_norm_curve, axis=0)
            succ_comp_score = np.mean(succ_comp_curve)
            prec_norm_score = np.mean(prec_norm_curve)    # AUC
            # prec_norm_score = prec_norm_curve[20]
            succ_comp_rate = succ_comp_curve[self.nbins_iou_complete // 2]

            if np.count_nonzero(speeds) > 0:
                avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]['overall'].update({
                'success_curve': succ_curve.tolist(),
                'precision_curve': prec_curve.tolist(),
                'success_complete_curve': succ_comp_curve.tolist(),
                'precision_norm_curve': prec_norm_curve.tolist(),
                'success_score': succ_score,
                'precision_score': prec_score,
                'success_rate': succ_rate,
                'success_complete_score': succ_comp_score,
                'precision_norm_score': prec_norm_score,
                'success_complete_rate': succ_comp_rate,
                'speed_fps': avg_speed})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)

        # plot precision and success curves for overall performance
        self.plot_curves(tracker_names, self.att_name)

        return performance


    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))

        play_speed = int(round(play_speed))
        assert play_speed > 0

        for s, seq_name in enumerate(seq_names):
            print('[%d/%d] Showing results on %s...' % (
                s + 1, len(seq_names), seq_name))

            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, '%s.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')

            # loop over the sequence and display results
            img_files, anno = self.dataset[seq_name]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [
                    records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    def _record(self, record_file, boxes, times):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('  Results recorded at', record_file)

        # record running times
        time_dir = os.path.join(record_dir, 'times')
        if not os.path.isdir(time_dir):
            os.makedirs(time_dir)
        time_file = os.path.join(time_dir, os.path.basename(
            record_file).replace('.txt', '_time.txt'))
        np.savetxt(time_file, times, fmt='%.8f')

    def _calc_metrics(self, boxes, anno):
        # can be modified by children classes
        ious = rect_iou(boxes, anno)                         # success plots
        center_errors = center_error(boxes, anno)            # precision plots

        ious_complete = rect_iou_complete(boxes, anno)       # complete success plots
        center_errors_norm = center_error_norm(boxes, anno)  # normalized precision plots

        return ious, center_errors, ious_complete, center_errors_norm

    def _calc_curves(self, ious, center_errors, ious_complete, center_errors_norm):
        ious = np.asarray(ious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]
        ious_complete = np.asarray(ious_complete, float)[:, np.newaxis]
        center_errors_norm = np.asarray(center_errors_norm, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]                      # 0, 0.05, 0.1, ..., 1
        thr_ce = np.arange(0, self.nbins_ce)[np.newaxis, :]                             # 0, 1, 2, 3, ..., 50
        thr_iou_complete = np.linspace(0, 1, self.nbins_iou_complete)[np.newaxis, :]    # 0, 0.05, 0.1, ..., 1
        thr_ce_norm = np.arange(0, self.nbins_ce_norm)[np.newaxis, :] / 100             # 0, 1, 2, 3, ..., 50 / 100

        bin_iou = np.greater(ious, thr_iou)                             # num of frames * 21 success rate: >IoU threshold
        bin_ce = np.less_equal(center_errors, thr_ce)                   # num of frames * 51 precision: <= center error
        bin_iou_complete = np.greater(ious_complete, thr_iou_complete)  # num of frames * 21 complete success rate: >IoU threshold
        bin_ce_norm = np.less_equal(center_errors_norm, thr_ce_norm)    # num of frames * 51 normalized precision: <= center error

        succ_curve = np.mean(bin_iou, axis=0)                 # 21
        prec_curve = np.mean(bin_ce, axis=0)                  # 51
        succ_comp_curve = np.mean(bin_iou_complete, axis=0)   # 21
        prec_norm_curve = np.mean(bin_ce_norm, axis=0)        # 51

        return succ_curve, prec_curve, succ_comp_curve, prec_norm_curve

    def plot_curves(self, tracker_names, att_name):
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        assert os.path.exists(report_dir), \
            'No results found. Run "tracker" first' \
            'before plotting curves.'
        report_file = os.path.join(report_dir, 'performance.json')
        assert os.path.exists(report_file), \
            'No reports found. Run "report" first' \
            'before plotting curves.'

        # load pre-computed performance
        with open(report_file) as f:
            performance = json.load(f)

        succ_file = os.path.join(report_dir, 'success_plots_'+att_name+'.png')
        prec_file = os.path.join(report_dir, 'precision_plots_'+att_name+'.png')
        succ_comp_file = os.path.join(report_dir, 'complete_success_plots_'+att_name+'.png')
        prec_norm_file = os.path.join(report_dir, 'normalized_precision_plots_'+att_name+'.png')

        ##########################*****************************************
        succ_file_pdf = os.path.join(report_dir, 'success_plots_'+att_name+'.pdf')
        prec_file_pdf = os.path.join(report_dir, 'precision_plots_'+att_name+'.pdf')
        succ_comp_file_pdf = os.path.join(report_dir, 'complete_success_plots_'+att_name+'.pdf')
        prec_norm_file_pdf = os.path.join(report_dir, 'normalized_precision_plots_'+att_name+'.pdf')
        key = 'overall'

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 3]

        if att_name == "All":
            title_name_suffix = " on UW-COT220"
        else:
            title_name_suffix = " - " + att_name.replace("_", " ")

        ##############################################################################
        # 1. sort trackers by success score
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # ####################################################
        ShowNum = min(30, len(tracker_names))    # Show top 21 methods
        tracker_names = tracker_names[0:ShowNum]
        # ####################################################

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots(figsize=(10,10.2), dpi=300)
        plt.tick_params(labelsize=22)
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve'],
                            markers[i % len(markers)],
                            linewidth=2)  ##########################*****************************************
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score']))
        # matplotlib.rcParams.update({'font.size': 14})
        # matplotlib.rcParams.update({'font.size': 7.4})
        matplotlib.rcParams.update({'font.size': 20})           # Top 12 methods
        # matplotlib.rcParams.update({'font.size': 10})           # Top 21 methods, font size 8.0
        # matplotlib.rcParams.update({'font.size': 4.32})        # Top 43 methods, font size 4.32

        ##########################*****************************************
        matplotlib.rcParams.update({'axes.titlesize': 24})
        matplotlib.rcParams.update({'axes.titleweight': 'black'})
        matplotlib.rcParams.update({'axes.labelsize': 22})

        # legend = ax.legend(lines, legends, loc='center left',
        #                    bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='center right')

        # matplotlib.rcParams.update({'font.size': 9})
        matplotlib.rcParams.update({'font.size': 22})
        plt.rcParams['axes.labelsize'] = 22
        ax.title.set_size(22)
        ax.xaxis.label.set_size(22)
        ax.yaxis.label.set_size(22)
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots'+title_name_suffix)
        # ax.grid(True)
        ax.grid(True, linestyle='-.')  ##########################*****************************************
        fig.tight_layout()

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)
        ##########################*****************************************
        fig.savefig(succ_file_pdf,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300,
                    format='pdf',
                    transparent=True)

        ##############################################################################
        # 2. sort trackers by precision score
        tracker_names = list(performance.keys())
        prec = [t[key]['precision_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # ####################################################
        ShowNum = min(30, len(tracker_names))    # Show top 21 methods
        tracker_names = tracker_names[0:ShowNum]
        # ####################################################

        # plot precision curves
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots(figsize=(10,10.2), dpi=300)
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_curve'],
                            markers[i % len(markers)],
                            linewidth=2)  ##########################*****************************************
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['precision_score']))
        # matplotlib.rcParams.update({'font.size': 7.4})
        # matplotlib.rcParams.update({'font.size': 14})
        matplotlib.rcParams.update({'font.size': 20})
        # matplotlib.rcParams.update({'font.size': 10.0})           # Top 21 methods, font size 8.0
        # matplotlib.rcParams.update({'font.size': 4.32})        # Top 43 methods, font size 4.32

        ##########################*****************************************
        matplotlib.rcParams.update({'axes.titlesize': 24})
        matplotlib.rcParams.update({'axes.titleweight': 'black'})
        matplotlib.rcParams.update({'axes.labelsize': 22})

        # legend = ax.legend(lines, legends, loc='center left',
        #                    bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='center left')

        # matplotlib.rcParams.update({'font.size': 9})
        matplotlib.rcParams.update({'font.size': 22})
        plt.rcParams['axes.labelsize'] = 22
        ax.xaxis.label.set_size(22)
        ax.yaxis.label.set_size(22)
        ax.set(xlabel='Location error threshold',
               ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1),
               title='Precision plots'+title_name_suffix)
        # ax.grid(True)
        ax.grid(True, linestyle='-.')  ##########################*****************************************
        fig.tight_layout()

        print('Saving precision plots to', prec_file)
        fig.savefig(prec_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)
        ##########################*****************************************
        fig.savefig(prec_file_pdf,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300,
                    format='pdf',
                    transparent=True)

        ##############################################################################
        # 3. sort trackers by complete success score
        tracker_names = list(performance.keys())
        succ = [t[key]['success_complete_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # ####################################################
        ShowNum = min(30, len(tracker_names))    # Show top 21 methods
        tracker_names = tracker_names[0:ShowNum]
        # ####################################################

        # plot complete success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou_complete)
        fig, ax = plt.subplots(figsize=(10,10.2), dpi=300)
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_complete_curve'],
                            markers[i % len(markers)],
                            linewidth=2)  ##########################*****************************************
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_complete_score']))
        # matplotlib.rcParams.update({'font.size': 7.4})
        matplotlib.rcParams.update({'font.size': 20})
        # matplotlib.rcParams.update({'font.size': 10.0})           # Top 21 methods, font size 8.0
        # matplotlib.rcParams.update({'font.size': 4.32})        # Top 43 methods, font size 4.32

        ##########################*****************************************
        matplotlib.rcParams.update({'axes.titlesize': 24})
        matplotlib.rcParams.update({'axes.titleweight': 'black'})
        matplotlib.rcParams.update({'axes.labelsize': 22})

        # legend = ax.legend(lines, legends, loc='center left',
        #                    bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='center right')

        # matplotlib.rcParams.update({'font.size': 9})
        matplotlib.rcParams.update({'font.size': 22})
        plt.rcParams['axes.labelsize'] = 22
        ax.xaxis.label.set_size(22)
        ax.yaxis.label.set_size(22)
        ax.set(xlabel='Overlap threshold',
               ylabel='Complete success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Complete success plots'+title_name_suffix)
        # ax.grid(True)
        ax.grid(True, linestyle='-.')  ##########################*****************************************
        fig.tight_layout()

        print('Saving complete success plots to', succ_comp_file)
        fig.savefig(succ_comp_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)
        ##########################*****************************************
        fig.savefig(succ_comp_file_pdf,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300,
                    format='pdf',
                    transparent=True)

        ##############################################################################
        # 4. sort trackers by normalized precision score
        tracker_names = list(performance.keys())
        prec = [t[key]['precision_norm_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # ####################################################
        ShowNum = min(30, len(tracker_names))    # Show top 21 methods
        tracker_names = tracker_names[0:ShowNum]
        # ####################################################

        # plot normalized precision curves
        thr_ce = np.arange(0, self.nbins_ce_norm) / 100
        fig, ax = plt.subplots(figsize=(10,10.2), dpi=300)
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_norm_curve'],
                            markers[i % len(markers)],
                            linewidth=2)  ##########################*****************************************
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['precision_norm_score']))
        # matplotlib.rcParams.update({'font.size': 7.4})
        matplotlib.rcParams.update({'font.size': 20})
        # matplotlib.rcParams.update({'font.size': 10.0})           # Top 21 methods, font size 8.0
        # matplotlib.rcParams.update({'font.size': 4.32})        # Top 43 methods, font size 4.32

        ##########################*****************************************
        matplotlib.rcParams.update({'axes.titlesize': 24})
        matplotlib.rcParams.update({'axes.titleweight': 'black'})
        matplotlib.rcParams.update({'axes.labelsize': 22})

        # legend = ax.legend(lines, legends, loc='center left',
        #                    bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='center left')

        # matplotlib.rcParams.update({'font.size': 9})
        matplotlib.rcParams.update({'font.size': 22})
        plt.rcParams['axes.labelsize'] = 22
        ax.xaxis.label.set_size(22)
        ax.yaxis.label.set_size(22)
        ax.set(xlabel='Location error threshold',
               ylabel='Normalized precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1),
               title='Normalized precision plots'+title_name_suffix)
        # ax.grid(True)
        ax.grid(True, linestyle='-.')  ##########################*****************************************
        fig.tight_layout()

        print('Saving normalized precision plots to', prec_norm_file)
        fig.savefig(prec_norm_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)
        ##########################*****************************************
        fig.savefig(prec_norm_file_pdf,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300,
                    format='pdf',
                    transparent=True)

        # clear figure
        plt.cla()
        plt.close("all")
