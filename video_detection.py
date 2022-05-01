# usage python video_detection.py --video_path = path/to/testing/video/dir/

# update weight path and dataset path in mrcnn.visualize_cv2

import cv2
import numpy as np
import os
import sys
from mrcnn import parse_args

args = parse_args.parse_args()

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

from mrcnn.visualize_cv2 import model, display_instances, class_names

VIDEO_PATH = args.video

def convert(millis): 
    millis = int(millis)
    seconds = (millis / 1000) % 60
    seconds = int(seconds)
    millis = millis - seconds * 1000
    minutes = (millis / (1000 * 60)) % 60
    minutes = int(minutes)
    #hours = (millis/(1000 * 60 * 60)) % 24
    millis = millis %1000
    return "%02d:%02d:%04d" % (minutes, seconds, millis)


for names in sorted(os.listdir(VIDEO_PATH)):
    
    if names.endswith(('.mp4','.avi')):
        video_count +=  1
        count  = 0
        print("Video:", names)
        video = cv2.VideoCapture(os.path.join(VIDEO_PATH, names))
        # FPS = video.get(cv2.CAP_PROP_FPS)
        FPS = 25

        ret =  1
        check_pos  = 0
        check_nev   = 0
    
        while(ret):
            ret, frame = video.read()
            if not ret: break
            count += 1
            if count % (FPS * 5) == 0:          #second =  5
                print("     frame %d - " %count, convert(video.get(cv2.CAP_PROP_POS_MSEC)),"ms") 
                results = model.detect([frame], verbose=1)
                r = results[0]
    
                if r["rois"].shape[0]:      #detected
                    check_pos += 1
                else: check_nev += 1
        if (check_pos >= 3):
            video_pos_3_frame += 1
        video.release()
        cv2.destroyAllWindows()

print("Total videos: " ,video_count, "\n")
file.write("Positive videos 3 frames: ", video_positive, "\n")
file.write("Negative videos 3 frames: ", video_count - video_positive,"\n")
file.close()
