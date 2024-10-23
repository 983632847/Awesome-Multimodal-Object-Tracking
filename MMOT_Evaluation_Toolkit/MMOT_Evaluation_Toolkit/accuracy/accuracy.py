"""
baseline for 1st Anti-UAV
https://anti-uav.github.io/
Qiang Wang
2020.02.16
"""
from __future__ import absolute_import
import os
import glob
import json
import cv2
import numpy as np



def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, w1_1, h1_1) = bbox1
    (x0_2, y0_2, w1_2, h1_2) = bbox2
    x1_1 = x0_1 + w1_1
    x1_2 = x0_2 + w1_2
    y1_1 = y0_1 + h1_1
    y1_2 = y0_2 + h1_2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def not_exist(pred):
    return (len(pred) == 1 and pred[0] == 0) or len(pred) == 0


def eval(out_res, label_res, exist):
    measure_per_frame = []
    for _pred, _gt, _exist in zip(out_res, label_res, exist):
        measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred, _gt) if len(_pred) > 1 else 0)
    return np.mean(measure_per_frame)


def main(Tracker_Name, video_paths, video_num, output_dir, visulization=False):
    # setup experiments
    # video_paths = glob.glob(os.path.join('F:/WebUAV_Evaluation_Toolkit/datasets/WebUAV', '*'))
    # video_num = len(video_paths)
    # output_dir = os.path.join('F:/WebUAV_Evaluation_Toolkit/toolkit/results/WebUAV', 'Tracker')

    # report performance
    overall_performance = []
    for video_id, video_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(video_path)

        # load groundtruth
        res_file = os.path.join(video_path, 'groundtruth_rect.txt')
        with open(res_file, 'r') as f:
            label_res = f.readlines()
        if ',' in label_res[0]:
            label_res = [list(map(int, box.split(','))) for box in label_res]
        else:
            label_res = [list(map(int, box.split())) for box in label_res]
        # label_res = [np.array(box) - [1, 1, 0, 0] for box in label_res]
        # init_rect = label_res[0]

        # make exist label
        exist = []
        for idx, gt in enumerate(label_res, start=1):
            if gt[0] == 0 and gt[1] == 0 and gt[2] == 0 and gt[3] == 0:
                exist.append(0)
            else:
                exist.append(1)

        # load images
        if visulization:
            image_path = video_path + '/img/*.jpg'
            image_files = sorted(glob.glob(image_path))

        # load prediction
        out_file = os.path.join(output_dir, video_name+'.txt')
        with open(out_file, 'r') as ff:
            out_res = ff.readlines()
        if ',' in out_res[0]:
            out_res = [list(map(float, box.split(','))) for box in out_res]
        else:
            out_res = [list(map(float, box.split())) for box in out_res]

        #######################################################
        if len(out_res) != len(label_res):
            # print(video_path, "Length Error!")
            minLen = min(len(out_res), len(label_res))
            out_res = out_res[0:minLen]
            label_res = label_res[0:minLen]
            exist = exist[0:minLen]
        #######################################################

        if visulization:
            for frame_id, image_file in enumerate(image_files):
                frame = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)  # h*w*c
                _gt = label_res[frame_id]
                _exist = exist[frame_id]
                out = out_res[frame_id]
                if _exist:
                    cv2.rectangle(frame, (int(_gt[0]), int(_gt[1])), (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])),(0, 255, 0))
                cv2.putText(frame, 'exist' if _exist else 'not exist',
                            (frame.shape[1] // 2 - 20, 30), 1, 2, (0, 255, 0) if _exist else (0, 0, 255), 2)

                cv2.putText(frame, '#'+str(frame_id), (20, 40), 1, 2, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(out[0]), int(out[1])), (int(out[0] + out[2]), int(out[1] + out[3])),
                              (0, 255, 255))
                cv2.imshow(video_name, frame)
                cv2.waitKey(1)

                frame_id += 1

        if visulization:
            cv2.destroyAllWindows()

        mixed_measure = eval(out_res, label_res, exist)
        overall_performance.append(mixed_measure)
        # print('[%03d/%03d] %20s ACC: %.03f' % (video_id, video_num, video_name, mixed_measure))

    print('%s: [Overall] Mean ACC: %.03f\n' % (Tracker_Name, np.mean(overall_performance)))


if __name__ == '__main__':
    # Trackers_Name = ["SiamFC"]
    # Trackers_Name = ["SiamFC", "SiamRPN","ATOM", "MDNet","GOTURN"]   # retraining
    Trackers_Name = [
        "SiamFC", "ACT", "AlphaRefine", "ARCF", "ATOM",
        "AutoMatch", "AutoTrack", "BACF", "CACF", "CCOT",
        "CF2", "DaSiamRPN", "DeepSRDCF", "DiMP", "DSiam",
        "ECO", "GOTURN", "HiFT", "KCF", "KeepTrack",
        "LADCF", "LightTrack", "MCCT", "MDNet", "MetaTracker",
        "Ocean", "PrDiMP", "SiamBAN", "SiamCAR", "SiamDW",
        "SiamFCpp", "SiamGAT", "SiamMask", "SiamRPN", "SiamRPNpp",
        "STRCF", "TransT", "TrDiMP", "UDT", "UpdateNet",
        "UTrack", "VITAL", "RPT"]


    for i in range(len(Trackers_Name)):
        Tracker_Name = Trackers_Name[i]
        video_paths = glob.glob(os.path.join('D:/WebUAV_demo/TPAMI_Evaluation/Test_Set', '*'))         # Benchmark Dataset GT
        video_num = len(video_paths)
        output_dir = os.path.join('F:/WebUAV_Evaluation_Toolkit/toolkit/results/Second_Name_TPAMI/WebUAV-3M', Tracker_Name)         # Results: Test
        # output_dir = os.path.join('F:/WebUAV_Evaluation_Toolkit/toolkit/results/Second_Name_TPAMI/Results_Retraining', Tracker_Name)  # Results: Retraining

        main(Tracker_Name, video_paths, video_num, output_dir, visulization=False)      # Accuracy



    # ######################################################################################
    # video_paths = glob.glob(os.path.join('D:/WebUAV_demo/TPAMI_Evaluation/Test_Set', '*'))
    # video_num = len(video_paths)
    # output_dir = os.path.join('I:/UAV_Tracking/Scripts/Toolkit_Evaluation/Scale/Scale-SiamRPN/WebUAV-3M-Scale80')  # Results: Test
    # main("Tracker_Name", video_paths, video_num, output_dir, visulization=False)  # Accuracy
