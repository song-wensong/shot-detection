import cv2
import sys
import os
import shutil
from matplotlib import pyplot as plt
import numpy as np

frames = []
frames_distance = []
frames_index = []
frames_edge = []

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
    last_frame = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    max_diff = 0
    dilate_rate = 5

    while True:
        # frame is a valid image if and only if ret is true
        (ret, frame) = cap.read()
        if not ret:
            break
        
        temp_frame = Frame()
        # the index of frame is from 0
        temp_frame.index = index
        temp_frame.img = frame.copy()
        
        # if last_frame is not None:
        # The images are converted to grayscale images;
        # for further processing, only the brightness of the pixels is important.
        # This reduces the need for storage and shortens the processing time.
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        # A and B are converted into edge images A' and B'. The conversion is usually done by Canny Edge detector.
        edge = cv2.Canny(gray_image, 0, 200)
        edge2 = cv2.Canny(gray_image2, 0, 200)
        frames_edge.append(edge)
        # From A' and B', the dilated images A* and B* are calculated.
        # Dilation widens the visible outlines. The optimal value for the parameter of dilatation is determined by empirical series of experiments.
        dilated = cv2.dilate(edge, np.ones((dilate_rate, dilate_rate)))
        dilated2 = cv2.dilate(edge2, np.ones((dilate_rate, dilate_rate)))
        # an inverted version of image dilated
        inverted = (255 - dilated)
        # an inverted version of image dilated2
        inverted2 = (255 - dilated2)
        # AND
        log_and1 = (edge2 & inverted)
        log_and2 = (edge & inverted2)
        # the number of all colored pixels is counted.
        pixels_sum_new = np.sum(edge)
        pixels_sum_old = np.sum(edge2)
        # incoming edge pixels and escaping edge pixels
        out_pixels = np.sum(log_and1)
        in_pixels = np.sum(log_and2)
        safe_div = lambda x,y: 0 if y == 0 else x / y
        # the maximum of edge entry ratio
        difference = max(safe_div(float(in_pixels),float(pixels_sum_new)), safe_div(float(out_pixels),float(pixels_sum_old)))

        # difference = 0
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
    print("cv2.CAP_PROP_POS_FRAMES = %d\n" %(frame_count))

    # threshold = max_diff * 3 / 4
    # threshold = max_diff / 2
    threshold = 0.31
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
        if frames[i].distance >= threshold:
            num += 1
            cv2.imwrite(shot_result_path + "/" + str(frames[i].index) + ".png", frames[i].img)
            cv2.imwrite(shot_result_path + "/" + str(frames[i].index) + "_diff.png", frames_edge[i])
    
    print("the number of shots dectected: %d\n" %(num))
    plt.figure()
    plt.title("edge change ratio")
    plt.xlabel("frame number")
    plt.ylabel("ECR")
    plt.plot(frames_index, frames_distance)
    plt.savefig("ecr.png")

    # Compute runtime and average framerate
    total_runtime = float(cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    avg_framerate = float(frame_count) / total_runtime

    print("Read %d frames from video in %4.2f seconds (avg. %4.1f FPS)." % (
        frame_count, total_runtime, avg_framerate))

    cap.release()

    
        

if __name__ == "__main__":
    main()