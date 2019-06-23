# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\ComputerVision\parking\ui.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
from twilio.rest import Client

# Configuration that will be used by the Mask-RCNN library
account_sid = 'ACe86da43edb0a15fe3119b548936e2d67'
auth_token = 'abb6d35eb495241ff955ff65e1da0d66'
client = Client(account_sid, auth_token)

class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(588, 539)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 531, 51))
        font = QtGui.QFont()
        font.setPointSize(26)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.lbl_img = QtWidgets.QLabel(self.centralwidget)
        self.lbl_img.setGeometry(QtCore.QRect(350, 260, 211, 211))
        self.lbl_img.setText("")
        self.lbl_img.setPixmap(QtGui.QPixmap("searching.jpg"))
        self.lbl_img.setScaledContents(True)
        self.lbl_img.setObjectName("lbl_img")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(40, 310, 251, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.lbl_slot = QtWidgets.QLabel(self.centralwidget)
        self.lbl_slot.setGeometry(QtCore.QRect(290, 320, 47, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lbl_slot.setFont(font)
        self.lbl_slot.setObjectName("lbl_slot")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(180, 120, 221, 121))
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 588, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.pushButton.clicked.connect(self.search_parking_space)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "FIND AVAILABLE PARKING SPACE"))
        self.label_3.setText(_translate("MainWindow", "Available parking slot on:"))
        self.lbl_slot.setText(_translate("MainWindow", "-"))
        self.pushButton.setText(_translate("MainWindow", "START"))
    
    def search_parking_space(self):
        # Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
        def get_car_boxes(boxes, class_ids):
            car_boxes = []

            for i, box in enumerate(boxes):
                # If the detected object isn't a car / truck, skip it
                if class_ids[i] in [3, 8, 6]:
                    car_boxes.append(box)

            return np.array(car_boxes)

        # Root directory of the project
        ROOT_DIR = Path(".")

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

        # Directory of images to run detection on
        IMAGE_DIR = os.path.join(ROOT_DIR, "images")

        # Video file or camera to process - set this to 0 to use your webcam instead of a video file
        VIDEO_SOURCE = "test_images/parking.mp4"

        # Create a Mask-RCNN model in inference mode
        model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

        # Load pre-trained model
        model.load_weights(COCO_MODEL_PATH, by_name=True)

        # Location of parking spaces
        parked_car_boxes = None

        # How many frames of video we've seen in a row with a parking space open
        free_space_frames = 0

        # Have we sent an SMS alert yet?
        sms_sent = False

        # Load the video file we want to run detection on
        video_capture = cv2.VideoCapture(VIDEO_SOURCE)

        # Free parking space slot
        free_space_slot  = 0
        # Loop over each frame of video
        while video_capture.isOpened():
            success, frame = video_capture.read()
            if not success:
                break

            # Convert the image from BGR color (which OpenCV uses) to RGB color
            rgb_image = frame[:, :, ::-1]

            # Run the image through the Mask R-CNN model to get results.
            results = model.detect([rgb_image], verbose=0)

            # Mask R-CNN assumes we are running detection on multiple images.
            # We only passed in one image to detect, so only grab the first result.
            r = results[0]

            # The r variable will now have the results of detection:
            # - r['rois'] are the bounding box of each detected object
            # - r['class_ids'] are the class id (type) of each detected object
            # - r['scores'] are the confidence scores for each detection
            # - r['masks'] are the object masks for each detected object (which gives you the object outline)

            if parked_car_boxes is None:
                # This is the first frame of video - assume all the cars detected are in parking spaces.
                # Save the location of each car as a parking space box and go to the next frame of video.
                parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
            else:
                # We already know where the parking spaces are. Check if any are currently unoccupied.

                # Get where cars are currently located in the frame
                car_boxes = get_car_boxes(r['rois'], r['class_ids'])

                # See how much those cars overlap with the known parking spaces
                overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)

                # Assume no spaces are free until we find one that is free
                free_space = False

                # Loop through each known parking space box
                c = 0
                for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):
                    c += 1
                    # For this parking space, find the max amount it was covered by any
                    # car that was detected in our image (doesn't really matter which car)
                    max_IoU_overlap = np.max(overlap_areas)

                    # Get the top-left and bottom-right coordinates of the parking area
                    y1, x1, y2, x2 = parking_area

                    # Check if the parking space is occupied by seeing if any car overlaps
                    # it by more than 0.15 using IoU
                    if max_IoU_overlap < 0.15:
                        # Parking space not occupied! Draw a green box around it
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 2), 2)

                        # Capturing video
                        _, frame = video_capture.read()
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.imwrite('result.jpg', frame)
                        self.lbl_img.setPixmap(QtGui.QPixmap("result.jpg"))
                        self.lbl_slot.setText(str(c))
                        # Clean up everything when finished
                        video_capture.release()
                        cv2.destroyAllWindows()

                        # Flag that we have seen at least one open space
                        free_space = True
                        free_space_slot = c
                    else:
                        # Parking space is still occupied - draw a red box around it
                        cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 2), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

                    # Write the IoU measurement inside the box
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))

                # If at least one space was free, start counting frames
                # This is so we don't alert based on one frame of a spot being open.
                # This helps prevent the script triggered on one bad detection.
                if free_space:
                    free_space_frames += 1
                else:
                    # If no spots are free, reset the count
                    free_space_frames = 0

                # If a space has been free for several frames, we are pretty sure it is really free!
                if free_space_frames > 2:
                    # If we haven't sent an SMS yet, sent it!
                    if not sms_sent:
                        print("SENDING SMS!!!")
                        print(free_space_slot)
                        sms_sent = True
                        
                        text_message = "Hello, parking spot number "+ str(free_space_slot) + " avaliable"
                        message = client.messages.create(
                            body= text_message,
                            from_='+14023474589',
                            to='+628988720006'
                        )

                # Show the frame of video on the screen
                cv2.imshow('Video', frame)

            # Hit 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Clean up everything when finished
        video_capture.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

