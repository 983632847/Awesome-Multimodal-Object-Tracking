from __future__ import absolute_import
import glob
import cv2
import os
import shutil
import numpy as np

def main(input_path, output_path):
    video_paths = glob.glob(os.path.join(input_path, "*"))
    output_dir = os.path.join(output_path)
    os.makedirs(output_dir, exist_ok=True)

    for video_id, video_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(video_path)
        video_dir = os.path.join(video_path, video_name + ".mp4")
        output_file = os.path.join(output_dir,  "%s" % video_name, "imgs")

        os.makedirs(output_file, exist_ok=True)
        print(video_id, video_name, "Addressing!")


        capture = cv2.VideoCapture(os.path.join(video_dir))
        FPS = capture.get(cv2.CAP_PROP_FPS)
        every_frame = int(FPS/FPS)   # save each frame

        frame_id = 1
        while True:
            ret, frame = capture.read()
            if not ret:
                capture.release()
                break

            ## Save frames
            if frame_id % every_frame ==0:
                save_image_path = os.path.join(output_file, "{}.jpg".format('%08d' % frame_id))
                if not os.path.exists(save_image_path):
                    cv2.imwrite(save_image_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            ## Draw first frame
            if frame_id == 1:
                init_output_file = os.path.join(output_dir, video_name+"_00000001.jpg")
                gt_path = os.path.join(video_path, "groundtruth_rect.txt")
                gts = np.loadtxt(gt_path, dtype=float, delimiter=',')
                gt = gts[0]
                draw_1 = cv2.rectangle(frame, (int(gt[0]), int(gt[1])), (int(gt[0] + gt[2]), int(gt[1] + gt[3])), (0, 0, 255), 2)
                cv2.imencode('.jpg', draw_1)[1].tofile(init_output_file)

            frame_id = frame_id + 1


if __name__ == "__main__":
    input_path = "home/WebUOT-1M/Test"
    output_path = "home/WebUOT-1M/Test"
    # input_path = "home/WebUOT-1M/Train"
    # output_path = "home/WebUOT-1M/Train"
    main(input_path, output_path)




