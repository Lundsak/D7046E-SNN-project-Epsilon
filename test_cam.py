import cv2
import numpy as np

bin_th = 20

vid = cv2.VideoCapture(1)
frame_count = 0
previous_frame = None

while (True):
    frame_count += 1

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    org_h, org_w = frame.shape[:2]

    # Resize the image using cv2.resize()
    frame = cv2.resize(frame, (100, org_h))

    if ((frame_count % 2) == 0):

        # 2. Prepare image; grayscale and blur
        prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prepared_frame = cv2.GaussianBlur(src=prepared_frame,
                                          ksize=(0, 0),
                                          sigmaX=1)

        # 3. Set previous frame and continue if there is None
        if (previous_frame is None):
            # First frame; there is no previous one yet
            previous_frame = prepared_frame
            continue

        # calculate difference and update previous frame
        diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
        previous_frame = prepared_frame

        # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)
        # cv2.imshow('Diff Image', diff_frame)

        # Sum the columns of the matrix
        sum_cols = cv2.reduce(diff_frame, 0, cv2.REDUCE_AVG)
        thresh_frame = cv2.threshold(src=sum_cols,
                                     thresh=bin_th,
                                     maxval=255,
                                     type=cv2.THRESH_BINARY)[1]

        # Resize the image using cv2.resize()
        resized_img = cv2.resize(thresh_frame, (org_w, 100))
        cv2.imshow('Movement', resized_img)

        
        
        # cv2.imshow('Movement', thresh_frame)

        


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()