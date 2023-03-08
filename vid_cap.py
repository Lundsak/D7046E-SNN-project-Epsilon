import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

verbose = False
show_plot = False
show_frames = True

frames_dir = "frames"
frame_count = 0

def ResizeDisplayAndSaveFrame(win_name, frame, size = None):
    if size is None:
        resized = frame
    else:
        resized = cv2.resize(frame, size)
    cv2.imshow(win_name, resized)
    cv2.imwrite(os.path.join(frames_dir, f"{win_name}-{frame_count}.bmp"),
                    resized)
    return resized



if not os.path.isdir(frames_dir):
    os.makedirs(frames_dir)

bin_th = 5
num_neurons = 100

vid = cv2.VideoCapture(1)

previous_frame = None

recording = False
start_time = 0
spikeArr = []
timeArr = []

while (True):

    frame_count += 1

    # Capture the video frame by frame
    ret, frame = vid.read()
    org_h, org_w = frame.shape[:2]

    # Resize to half so that the main frame does not occupy the entire screen
    # and flip it to create a mirror effect.
    org_w = int(org_w / 2)
    org_h = int(org_h / 2)
    frame = cv2.resize(frame, (org_w, org_h))
    if show_frames:
        ResizeDisplayAndSaveFrame("Main", frame)
    
    frame = cv2.flip(frame, 1)
    if show_frames:
        ResizeDisplayAndSaveFrame("Mirrored", frame)

    if ((frame_count % 2) == 0):
        # Resize the width to the number of neurons
        frame = cv2.resize(frame, (num_neurons, org_h), cv2.INTER_CUBIC)
        if show_frames:
            ResizeDisplayAndSaveFrame("Resized", frame, (org_w, org_h))

        # Prepare image; grayscale and blur
        prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if show_frames:
            ResizeDisplayAndSaveFrame("Grayscale", prepared_frame, (org_w, org_h))

        prepared_frame = cv2.GaussianBlur(src=prepared_frame,
                                          ksize=(0, 0),
                                          sigmaX=3)
        if show_frames:
            ResizeDisplayAndSaveFrame("Blurred", prepared_frame, (org_w, org_h))

        # Set previous frame and continue if there is None
        if (previous_frame is None):
            # First frame; there is no previous one yet
            previous_frame = prepared_frame
            continue

        # Calculate difference and update previous frame
        diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
        if show_frames:
            ResizeDisplayAndSaveFrame("Diff", diff_frame, (org_w, org_h))

        previous_frame = prepared_frame

        # Sum the columns of the matrix
        sum_cols = cv2.reduce(diff_frame, 0, cv2.REDUCE_AVG)
        if show_frames:
            ResizeDisplayAndSaveFrame("1D Diff", sum_cols, (org_w, 100))

        thresh_frame = cv2.threshold(src=sum_cols,
                                     thresh=bin_th,
                                     maxval=255,
                                     type=cv2.THRESH_BINARY)[1]

        if show_frames:
            ResizeDisplayAndSaveFrame("1D Bin", thresh_frame, (org_w, 100))

        if not recording and 255 in thresh_frame:
            recording = True
            spikeArr = []
            timeArr = []
            start_time = time.time()

        if recording and not 255 in thresh_frame:
            recording = False
            if verbose:
                print('Spikes:', spikeArr)
                print('Times:', timeArr)
            # HERE WE SHOULD SEND spikeArr and timeArr TO THE NETWORK
            if show_plot:
                plt.scatter(timeArr, spikeArr)
                plt.xlabel('Time (s)')
                plt.ylabel('Neuron ID')
                plt.grid(True)
                plt.show()

        if recording:
            new_spikes = np.where(thresh_frame == 255)[1]
            new_times = np.ones(new_spikes.shape) * time.time() - start_time
            spikeArr.extend(new_spikes.tolist())
            timeArr.extend(new_times.tolist())
            # print('New Spikes:', new_spikes)
            # print('New Times:', new_times)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()