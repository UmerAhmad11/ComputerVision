#Modules to import

import matplotlib as plt
import numpy as np 
import tensorflow as tf 
import cv2 as cv


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot Open Camera")
        exit()
    while True:
        try:
            ret, frame = cap.read()

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Display the resulting frame
            cv.imshow('frame', gray)
            if cv.waitKey(1) == ord('q'):
                break

        except Exception as e:
            print("Error:", e)
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


#Run the main function
if __name__ == "__main__":
    main()