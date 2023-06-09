import cv2
import sys
import os
import shutil
from matplotlib import pyplot as plt
import numpy as np

frames = []
frames_distance = []
frames_index = []

class Frame:
    index = -1
    img = None
    distance = 0

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
    last_frame = 0
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
        
        # do guassian blur
        frame = cv2.GaussianBlur(frame, (9, 9), 0.0)
        # Sum of absolute differences
        difference = np.sum(np.absolute(frame - last_frame))
        temp_frame.distance = difference

        frames_distance.append(difference)
        frames_index.append(index)
        frames.append(temp_frame)
        
        # get the max distance
        if difference > max_diff:
            max_diff = difference

        index += 1
        last_frame = frame

    print("max_diff = %d, frame_count = %d\n" %(max_diff, index))



    # Get # of frames in video based on the position of the last frame we read.
    frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(frame_count)

    threshold = max_diff * 3 / 4
    # threshold = max_diff * 4 / 5
    # threshold = max_diff / 2
    # threshold = 0
    # threshold = max_diff
    # shot_result_path = "shot_result"
    # shot_result_path = "all_frame"
    # shot_result_path = "all_frame2"
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
        if frames[i].distance >= threshold:
            num += 1
            # print(frames[i].index)
            cv2.imwrite(shot_result_path + "/" + str(frames[i].index) + ".png", frames[i].img)

    print("the number of shots dectected: %d\n" %(num))
    plt.figure()
    plt.title("Sum of absolute differences")
    plt.xlabel("frame number")
    plt.ylabel("SAD")
    plt.plot(frames_index, frames_distance)
    plt.savefig("SAD.png")
    plt.show()

    # Compute runtime and average framerate
    total_runtime = float(cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    avg_framerate = float(frame_count) / total_runtime

    print("Read %d frames from video in %4.2f seconds (avg. %4.1f FPS)." % (
        frame_count, total_runtime, avg_framerate))

    cap.release()

    
        

if __name__ == "__main__":
    main()