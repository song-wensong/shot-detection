import cv2
import sys
import os
import shutil
from matplotlib import pyplot as plt
import numpy as np

frames = []
frames_distance = []
frames_correl = []
frames_bhattacharyya = []
frames_index = []

class Frame:
    index = -1
    img = None
    distance = 0
    correl = 0
    bhattacharyya = 0



def main():
    if len(sys.argv) < 2:
        print("Error - file name must be specified as first argument")
        return
    # Default constructor.
    cap = cv2.VideoCapture()
    # Open video file for video capturing.
    cap.open(sys.argv[1])
    # Returns true if video capturing has been initialized already.
    if not cap.isOpened():
        print("Fatal error - could not open video %s." % sys.argv[1])
        return
    else:
        print("Parsing video %s..." % sys.argv[1])
    
    # get the frame width and height
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Video Resolution: %d * %d" % (width, height))
    
    # initialization of some variables
    threshold = 0
    start_time = cv2.getTickCount()  # Used for benchmarking/statistics after loop.
    index = 0
    last_hist_h = np.zeros((180, 1), dtype = np.float32)
    last_hist_s = np.zeros((256, 1), dtype = np.float32)
    last_hist_v = np.zeros((256, 1), dtype = np.float32)
    max_diff = 0

    while True:
        # frame is a valid image if and only if ret is true
        (ret, frame) = cap.read()
        if not ret:
            break
        
        temp_frame = Frame()
        # the index of frame is from 0
        temp_frame.index = index
        temp_frame.img = frame.copy()
        
        # do gaussian blur
        frame = cv2.GaussianBlur(frame, (9, 9), 0.0)
        # convert an image from BGR color space to HSV color space
        frame= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # split coloured/multi-channel image into separate single-channel images
        h, s, v = cv2.split(frame)
        
        
        # calculate the image hue histograms
        hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
        cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # calculate the image saturation histograms
        hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
        cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # calculate the image saturation histograms
        hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
        cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        temp_frame.correl = cv2.compareHist(hist_h, last_hist_h, cv2.HISTCMP_CORREL) + cv2.compareHist(hist_s, last_hist_s, cv2.HISTCMP_CORREL) + cv2.compareHist(hist_v, last_hist_v, cv2.HISTCMP_CORREL)
        temp_frame.bhattacharyya = (cv2.compareHist(hist_h, last_hist_h, cv2.HISTCMP_BHATTACHARYYA) + cv2.compareHist(hist_s, last_hist_s, cv2.HISTCMP_BHATTACHARYYA) + cv2.compareHist(hist_v, last_hist_v, cv2.HISTCMP_BHATTACHARYYA))
        # difference = cv2.compareHist(hist_h, last_hist_h, cv2.HISTCMP_CORREL) + cv2.compareHist(hist_s, last_hist_s, cv2.HISTCMP_CORREL) + cv2.compareHist(hist_v, last_hist_v, cv2.HISTCMP_CORREL) - (cv2.compareHist(hist_h, last_hist_h, cv2.HISTCMP_BHATTACHARYYA) + cv2.compareHist(hist_s, last_hist_s, cv2.HISTCMP_BHATTACHARYYA) + cv2.compareHist(hist_v, last_hist_v, cv2.HISTCMP_BHATTACHARYYA))
        difference = temp_frame.correl - temp_frame.bhattacharyya
        # difference = 0
        temp_frame.distance = difference

        frames_distance.append(difference)
        frames_bhattacharyya.append(temp_frame.bhattacharyya)
        frames_correl.append(temp_frame.correl)
        frames_index.append(index)

        frames.append(temp_frame)
        
        # get the max distance
        if difference > max_diff:
            max_diff = difference

        index += 1
        last_hist_h = hist_h
        last_hist_s = hist_s
        last_hist_v = hist_v

    print("max_diff = %d, frame_count = %d\n" %(max_diff, index))
    

    # Get # of frames in video based on the position of the last frame we read.
    frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print("cv2.CAP_PROP_POS_FRAMES = %d\n" %(frame_count))

    # threshold = max_diff * 3 / 4
    threshold = 2.3
    # threshold = 1.5
    # threshold = 0
    # threshold = max_diff
    # shot_result_path = "shot_result"
    # shot_result_path = "all_frame"
    # shot_result_path = "histogram_diff_res"
    shot_result_path = sys.argv[2]
    

    # create the result directory
    folder = os.path.exists(shot_result_path)
    if not folder:
        os.makedirs(shot_result_path)
    else:
        shutil.rmtree(shot_result_path)
        os.makedirs(shot_result_path)
    
    num = 0
    for i in range(0, len(frames) - 1):
        # if i == 2:
        #     print(frames[i].img)
        if frames[i].distance <= threshold:
            num += 1
            # print(frames[i].index)
            cv2.imwrite(shot_result_path + "/" + str(frames[i].index) + ".png", frames[i].img)

    print("the number of shots dectected: %d\n" %(num))
    plt.figure()
    plt.title("histogram differences")
    plt.xlabel("frame number")
    plt.ylabel("histogram correl")
    plt.plot(frames_index, frames_correl)
    plt.savefig("histogram_correl.png")

    plt.figure()
    plt.title("histogram differences")
    plt.xlabel("frame number")
    plt.ylabel("histogram bhattacharyya")
    plt.plot(frames_index, frames_bhattacharyya)
    plt.savefig("histogram_bhattacharyya.png")

    plt.figure()
    plt.title("histogram differences")
    plt.xlabel("frame number")
    plt.ylabel("histogram distance")
    plt.plot(frames_index, frames_distance)
    plt.savefig("histogram_distance.png")

    # Compute runtime and average framerate
    total_runtime = float(cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    avg_framerate = float(frame_count) / total_runtime

    print("Read %d frames from video in %4.2f seconds (avg. %4.1f FPS)." % (
        frame_count, total_runtime, avg_framerate))

    cap.release()

    
        

if __name__ == "__main__":
    main()