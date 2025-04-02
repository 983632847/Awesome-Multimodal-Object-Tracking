# Video segmentation with SAM 2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import random
from torchvision.ops import masks_to_boxes
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
HYDRA_FULL_ERROR=1
import clip

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    plt.show()


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)



# pip install torch==2.3.1
# python=3.11.0
# pip install hydra-core --upgrade

'''
cd /mnt/sdb/zhangchunhui/SAM2/VL-SAM2/
conda activate SAM2 
nohup python Test_VL_SAM2_UWCOT220.py > output.log 2>&1 &
'''
if __name__=='__main__':
    device = "cuda"

    ###########################################
    ## Points
    Num_of_Points = 0
    # Num_of_Points = 1
    # Num_of_Points = 2
    # Num_of_Points = 3

    dataset_path = '/mnt/sdb/zhangchunhui/Datasets/Underwater/UWCOT220'
    output_path = '/mnt/sdb/zhangchunhui/SAM2/Results/UWCOT2022-SAM2.1L-CLIP-Frozen-image-encoder'

    os.makedirs(output_path, exist_ok=True)

    #############################################################
    ## Loading the SAM 2 video predictor
    sam2_checkpoint = "/mnt/sdb/zhangchunhui/SAM2/VL-SAM2/training/sam2_logs/configs-SAM2.1L-CLIP-Frozen-image-encoder/sam2l_YTVOS_finetune.yaml/checkpoints/checkpoint.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor: SAM2VideoPredictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    videos = sorted(os.listdir(dataset_path))
    for video_idx, video in enumerate(videos,start=1):
        predicted_result_path = os.path.join(output_path, video + ".txt")
        if os.path.exists(predicted_result_path):
            print(video_idx, video, 'Done!')
            continue
        else:
            print(video_idx, video, 'Doing!')

        #############################################################
        ## load GTs
        gt_path = os.path.join(dataset_path, video, 'groundtruth_rect.txt')
        with open(gt_path, 'r') as gt_file:
            gts = gt_file.readlines()
        gt = gts[0]
        gt = gt.strip('\n').split(',')  # xywh

        center_point = [int(float(gt[0])+float(gt[2]) / 2+0.5),
                        int(float(gt[1]) + float(gt[3]) / 2 + 0.5)]
        point2 = [int(float(gt[0])+random.randint(1, int(float(gt[2])-1))),
                        int(float(gt[1]) + random.randint(1, int(float(gt[3])-1)))]
        point3 = [int(float(gt[0])+random.randint(1, int(float(gt[2])-1))),
                        int(float(gt[1]) + random.randint(1, int(float(gt[3])-1)))]

        # bbox:[x1,y1,x2,y2]
        init_bbox = [int(float(gt[0])), int(float(gt[1])),
                     int(float(gt[0]) + float(gt[2])),
                     int(float(gt[1]) + float(gt[3]))]

        ## load language
        language_file = os.path.join(dataset_path, video, "language.txt")
        with open(language_file, 'r') as file:
            language = file.readline().replace(".", "").replace("\n", "").lower()

        #############################################################
        ## load video
        # video_dir = "notebooks/videos/bedroom"
        video_dir = os.path.join(dataset_path, video, 'imgs')

        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # # take a look the first video frame
        frame_idx = 0
        # plt.figure(figsize=(12, 8))
        # plt.title(f"frame {frame_idx}")
        initial_image = Image.open(os.path.join(video_dir, frame_names[frame_idx]))
        # plt.imshow(initial_image)
        # plt.show()

        num_images = os.listdir(os.path.join(dataset_path, video, 'imgs'))
        try:
            # Initialize the inference state
            inference_state = predictor.init_state(video_path=video_dir)
            # Segment & track one object
            predictor.reset_state(inference_state)
        except:  # if out of GPU memory (long videos), use cpu
            # Initialize the inference state
            inference_state = predictor.init_state(video_path=video_dir, offload_video_to_cpu=True)
            # Segment & track one object
            predictor.reset_state(inference_state)

        #############################################################
        ## Step 1: Add a first click on a frame
        if Num_of_Points==1:
            ann_frame_idx = 0  # the frame index we interact with
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

            # Let's add a positive click at (x, y) = (210, 350) to get started
            # points = np.array([[210, 350]], dtype=np.float32)
            points = np.array([center_point], dtype=np.float32)
            # for labels, `1` means positive click and `0` means negative click
            labels = np.array([1], np.int32)

            out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

            # # show the results on the current (interacted) frame
            # plt.figure(figsize=(12, 8))
            # plt.title(f"frame {ann_frame_idx}")
            # plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
            # show_points(points, labels, plt.gca())
            # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
            # plt.show()

        #############################################################
        ## Step 2: Add a second click to refine the prediction
        elif Num_of_Points==2:
            ann_frame_idx = 0  # the frame index we interact with
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

            # Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
            # sending all clicks (and their labels) to `add_new_points`
            # points = np.array([[210, 350], [250, 220]], dtype=np.float32)
            points = np.array([center_point, point2], dtype=np.float32)
            # for labels, `1` means positive click and `0` means negative click
            labels = np.array([1, 1], np.int32)
            out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

            # # show the results on the current (interacted) frame
            # plt.figure(figsize=(12, 8))
            # plt.title(f"frame {ann_frame_idx}")
            # plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
            # show_points(points, labels, plt.gca())
            # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

        #############################################################
        ## Step 3: Add a third click to refine the prediction
        elif Num_of_Points==3:
            ann_frame_idx = 0  # the frame index we interact with
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

            # Let's add a 3rd positive click at (x, y) = (250, 220) to refine the mask
            # sending all clicks (and their labels) to `add_new_points`
            # points = np.array([[210, 350], [250, 220], [260, 200]], dtype=np.float32)
            points = np.array([center_point, point2, point3], dtype=np.float32)
            # for labels, `1` means positive click and `0` means negative click
            labels = np.array([1, 1, 1], np.int32)
            out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

            # # show the results on the current (interacted) frame
            # plt.figure(figsize=(12, 8))
            # plt.title(f"frame {ann_frame_idx}")
            # plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
            # show_points(points, labels, plt.gca())
            # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        elif Num_of_Points==0:
            # Load the model
            model, _ = clip.load("ViT-B/32", device=device, jit=False)
            texts = [language]
            input_ids = clip.tokenize(texts, truncate=True).to(device)

            with torch.no_grad():
                lang_embedds = model.encode_text(input_ids)
                lang_embedds /= lang_embedds.norm(dim=-1, keepdim=True)
            # lang_embedds = torch.randn(1, 512, device=device)

            # add new prompts and instantly get the output on the same frame
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=init_bbox,    #######################
                lang_embedds=lang_embedds,
            )
        else:
            print("Wrong number of points, please give 1,2,or 3 points!")

        #############################################################
        ## Step 4: Propagate the prompts to get the masklet across the video
        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Render the segmentation results every few frames
        # vis_frame_stride = 15
        vis_frame_stride = 1
        # plt.close("all")
        New_Box = []
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            # plt.figure(figsize=(6, 4))
            # plt.title(f"frame {out_frame_idx}")
            # plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                # show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

                #####################################3
                ## Save binary mask
                if out_mask is not None:
                    # show_res(masks, scores, input_point, input_label, input_box, result_path,'demo', image)
                    output = out_mask.astype(np.uint8)  # H*W
                    h, w = output.shape[-2:]
                    mask_image = output.reshape(h, w, 1)  # H*W*1
                    mask_image = mask_image * 255  # 0: black, 255: white

                    ## Masks to box
                    try:
                        bbox_ = masks_to_boxes(torch.tensor(output)).numpy()
                        bbox_ = bbox_[0]
                        # xmin,ymin,xmax,ymax->xywh
                        refined_box = [bbox_[0], bbox_[1], bbox_[2] - bbox_[0], bbox_[3] - bbox_[1]]
                    except:
                        refined_box = [0, 0, 0, 0]
                    print(out_frame_idx, "Predicted bbox(xywh):", refined_box)
                    New_Box.append(refined_box)

                # Case3. target object mask is not be detected
                else:
                    h, w = initial_image.shape[0:2]
                    mask_image = np.zeros((h, w))  # black
                    # bbox: x1y1x2y2
                    # mask_image[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 255  # white
                    mask_image = mask_image.reshape(h, w, 1)

                    refined_box = [0, 0, 0, 0]
                    print(out_frame_idx, "No Predicted bbox(xywh):", refined_box)
                    New_Box.append(refined_box)

            # os.makedirs(os.path.join(output_path, video, 'imgs'), exist_ok=True)
            # output_image_path = os.path.join(output_path, video, 'imgs', frame_names[frame_idx].replace('.jpg', '.png'))
            # cv2.imwrite(output_image_path, mask_image)

        predicted_result_path = os.path.join(output_path, video+".txt")
        np.savetxt(predicted_result_path, np.array(New_Box), fmt='%d', delimiter=',')
