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
import matplotlib.ticker as ticker
import random
import matplotlib.pyplot as plt


def plot_ranking_curve(names, scores, highlight_ranks=None):
    """
    绘制学术风格的排名图
    highlight_ranks: 一个列表，包含要高亮标注的排名序号，例如 [1, 5, 10]
    """
    if highlight_ranks is None:
        highlight_ranks = [1, 5, 10]

    n = len(names)
    ranks = np.arange(n, 0, -1)  # 排名从 N 到 1

    # === 1. 设置绘图风格 ===
    # 设置字体（推荐使用Times New Roman或Arial用于学术论文）
    plt.rcParams['font.family'] = 'DejaVu Sans'  # 如果有Times New Roman建议替换
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.5  # 边框加粗

    # 定义大量的Marker和Color，用于区分50个算法
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '1', '2', '3', '4']
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#008000', '#FFD700',
              '#A52A2A']

    fig, ax = plt.subplots(figsize=(10, 9))  # 画布高一点，留给底部的Legend

    # === 2. 绘制数据点和连线 ===
    # 绘制连线 (虚线或实线)
    ax.plot(ranks, scores, color='blue', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

    # 绘制每个算法的散点
    scatter_handles = []
    for i in range(n):
        # 循环使用样式
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]

        # 绘制点
        # zorder确保点在线上层
        sc = ax.scatter(ranks[i], scores[i], marker=marker, color=color, s=80, edgecolors='black', linewidth=0.5,
                        label=names[i], zorder=2)
        scatter_handles.append(sc)

    # === 3. 坐标轴设置 ===
    ax.set_xlim(n + 1, 0)  # X轴反转！Rank 1 在右边
    ax.set_xlabel("Rank", fontsize=14, fontweight='bold')
    ax.set_ylabel("mACC Score", fontsize=14, fontweight='bold')

    # 设置刻度朝内 (Inward ticks) - 学术图表常见风格
    ax.tick_params(direction='in', length=6, width=1.5, top=True, right=True)

    # 设置X轴刻度间隔 (例如每5个显示一个)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    # === 添加基准线及Y轴数值 ===
    mean_score = np.mean(scores)
    # 画线
    ax.axhline(y=mean_score, color='gray', linestyle='-', linewidth=2, alpha=0.3)

    # 在Y轴上显示基准线的刻度值
    # 使用annotate定位到Y轴的左侧 (xy的x=0, axes fraction)，并向左偏移一定距离
    ax.annotate(f'{mean_score:.3f}',
                xy=(0, mean_score), xycoords=('axes fraction', 'data'),
                xytext=(-5, 0), textcoords='offset points',
                ha='right', va='center',
                color='gray', fontsize=9, fontweight='bold')

    # === 4. 高亮标注指定排名的算法 (水平箭头版) ===
    # 根据用户传入的 rank 列表进行标注
    for i, r in enumerate(highlight_ranks):
        # 检查排名是否有效
        if r < 1 or r > n:
            print(f"警告: 排名 {r} 超出范围 (1-{n})，已跳过")
            continue

        # 转换 rank 到 index
        # ranks数组是 [50, 49, ..., 1]，对应的 scores 是升序
        # Rank 1 是 scores 的最后一个 (index = n-1)
        # Rank r 对应的 index 是 n - r
        index = n - r

        score = scores[index]
        rank = ranks[index]
        name = names[index]

        # 添加水平箭头标注
        # xytext在点的左侧 (Rank数值更大处)，Y轴保持一致，实现水平效果
        # offset_rank 决定了箭头长度/文字距离
        offset_rank = 3.0

        ax.annotate(name,
                    xy=(rank, score),
                    xytext=(rank + offset_rank, score),  # Y轴不变，保证水平
                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                    fontsize=10, fontweight='bold',
                    ha='right',  # 文字右对齐，紧贴箭头尾部
                    va='center')  # 文字垂直居中，对齐箭头

    # === 5. 制作底部复杂图例 (更新：包含数值) ===
    # 准备带有数值的标签列表
    # names 和 scores 目前是升序 (Rank N -> Rank 1)
    # 我们需要反转它们，让 Rank 1 出现在图例的第一个位置
    sorted_names_desc = names[::-1]
    sorted_scores_desc = scores[::-1]

    # 格式化标签: "AlgorithmName [0.451]"
    legend_labels = [f"{name} [{score:.3f}]" for name, score in zip(sorted_names_desc, sorted_scores_desc)]

    # 将Legend放在图表正下方，分多列显示
    # args: handles, labels
    # ncol: 列数，根据算法数量调整，50个算法建议 5-8列
    legend = ax.legend(handles=scatter_handles[::-1],
                       labels=legend_labels,  # 使用带有数值的新标签
                       loc='upper center',
                       bbox_to_anchor=(0.5, -0.15),  # 放置在X轴下方
                       ncol=5,  # 5列，因为加了数值标签变宽了，列数稍微减少一点防止拥挤
                       frameon=False,  # 不要边框
                       fontsize=9,
                       columnspacing=0.8,
                       handletextpad=0.1)

    # 调整布局以适应底部的Legend
    plt.subplots_adjust(bottom=0.35)

    # 添加左上角的特殊符号 (如图片中的 Phi hat)
    ax.text(0.02, 0.98, r'$\hat{\ \Phi}$', transform=ax.transAxes, fontsize=24, va='top')  # 字符前加一点负间距（negative space）

    plt.grid(True, which='major', linestyle=':', alpha=0.6)

    # 保存 png 和 pdf
    plt.savefig('mACC_Ranking_Plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('mACC_Ranking_Plot.pdf', dpi=300, bbox_inches='tight', format='pdf')
    return fig

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
    # report performance
    overall_performance = []
    for video_id, video_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(video_path)

        # load groundtruth
        res_file = os.path.join(video_path, 'groundtruth_rect.txt')
        with open(res_file, 'r') as f:
            label_res = f.readlines()
        if ',' in label_res[0]:
            # if '.' in label_res[0]:
            try:
                label_res = [list(map(float, box.split(','))) for box in label_res]
            # else:
            except:
                label_res = [list(map(int, box.split(','))) for box in label_res]
        else:
            # if '.' in label_res[0]:
            try:
                label_res = [list(map(float, box.split())) for box in label_res]
            # else:
            except:
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
            image_path = video_path + '/imgs/*.jpg'
            image_files = sorted(glob.glob(image_path))

        # load prediction
        out_file = os.path.join(output_dir, video_name+'.txt')
        with open(out_file, 'r') as ff:
            out_res = ff.readlines()
        if ',' in out_res[0]:
            out_res = [list(map(float, box.split(','))) for box in out_res]
        else:
            out_res = [list(map(float, box.split())) for box in out_res]

        # #######################################################
        # if len(out_res) != len(label_res):
        #     # print(video_path, "Length Error!")
        #     minLen = min(len(out_res), len(label_res))
        #     out_res = out_res[0:minLen]
        #     label_res = label_res[0:minLen]
        #     exist = exist[0:minLen]
        # #######################################################

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

    print('%s: [Overall] Mean ACC: %.03f' % (Tracker_Name, np.mean(overall_performance)))
    mACC_Score = np.mean(overall_performance)
    return mACC_Score


if __name__ == '__main__':
    # Trackers_Name = ["All-in-One"]
    Trackers_Name = [
        "SiamFC", "ECO", "VITAL", "ATOM", "SiamMask",
        "SiamRPN++", "SiamFC++", "SiamBAN", "SiamCAR", "LightTrack",

        "SiamGAT", "TrDiMP", "TransT", "STARK-ST50", "KeepTrack",
        "HiFT", "AutoMatch", "TCTrack", "ToMP-101", "AiATrack",

        "SimTrack-B32", "OSTrack", "SGLATrack", "GRM", "ZoomTrack",
        "Aba-ViTrack", "HIPTrack", "AQATrack", "TCTrack++", "EVPTrack",

        "SeqTrack-B256", "DropTrack", "MixFormerV2", "ARTrackV2", "MCITrack-B224",
        "ODTrack", "LORAT-B224", "MambaNUT", "ORTrack", "MambaLCT",

        "ATCTrack", "JointNLT", "VLT_TT", "MambaTrack", "All-in-One",
        "CiteTracker-256", "UVLTrack", "DUTrack-256", "SUTrack-B224", "MambaSTS",
        ]

    mACC_Scores = []
    for i in range(len(Trackers_Name)):
        Tracker_Name = Trackers_Name[i]
        # Path of the Benchmark Dataset: GT
        video_paths = glob.glob(os.path.join('/mnt/Datasets/UAVAntiUAV_Tracking/Test', '*'))
        video_num = len(video_paths)

        # Results: Test
        output_dir = os.path.join('results/Baseline_Results/UAVAntiUAV', Tracker_Name)

        # Calculate Accuracy
        mACC_Score = main(Tracker_Name, video_paths, video_num, output_dir, visulization=False)
        mACC_Scores.append(mACC_Score)


    ###################################################################################
    ## Draw mACC ranking curve
    algo_names, algo_scores = Trackers_Name, mACC_Scores

    # If your data is out of order, you can do a simple sorting first (optional).
    # 函数 plot_ranking_curve 内部假设传入的是按分数*从小到大*排序的列表。如果你的数据是乱序的，请先做个简单的排序（可选）
    rerank = True
    if rerank:
        combined = sorted(zip(algo_names, algo_scores), key=lambda x: x[1])
        algo_names = [x[0] for x in combined]
        algo_scores = [x[1] for x in combined]


    # Specify the rank (horizontal arrow) to which you want to add arrows. 指定需要添加箭头的排名 (水平箭头)
    target_ranks = [1, 2, 4, 8, 11, 15, 20, 23, 26, 29]

    fig = plot_ranking_curve(algo_names, algo_scores, highlight_ranks=target_ranks)

    print("mACC_Ranking_Plot.png and mACC_Ranking_Plot.pdf done!")

