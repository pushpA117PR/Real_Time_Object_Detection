# How to run?: python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# python real_time.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import packages
from imutils.video import VideoStream
#Manages and captures video frames from a camera.
from imutils.video import FPS
#Measures and tracks the frames per second (FPS) in a video stream.
import numpy as np
#Provides support for numerical operations and array manipulations
import argparse
# Parses and handles command-line arguments.
import imutils
#Offers utility functions for image and video processing tasks
import time
#Provides functions to work with time, including delays and elapsed time.
import cv2
#Provides OpenCV functions for image and video processing.

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak predictions")
args = vars(ap.parse_args())
# Let's start by initialising the list of the 21 class labels MobileNet SSD was trained to.
# Each prediction composes of a boundary box and 21 scores for each class (one extra class for no object),
# and we pick the highest score as the class for the bounded object
CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor","pen","book","idcard","specticles"]

# Assigning random colors to each of the classes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# COLORS: a list of 21 R,G,B values, like ['101.097383   172.34857188 111.84805346'] for each label
# length of COLORS = length of CLASSES = 21

# load our serialized model
# The model from Caffe: MobileNetSSD_deploy.prototxt.txt; MobileNetSSD_deploy.caffemodel;
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# print(net)
# <dnn_Net 0x128ce1310>

# initialize the video stream,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# warm up the camera for a couple of seconds
time.sleep(2.0)

# FPS: used to compute the (approximate) frames per second
# Start the FPS timer
fps = FPS().start()
while True:
	# grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
	# vs is the VideoStream
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	print(frame.shape) # (225, 400, 3)
	# grab the frame dimensions and convert it to a blob
	# First 2 values are the h and w of the frame. Here h = 225 and w = 400
	(h, w) = frame.shape[:2]
	# Resize each frame
	resized_image = cv2.resize(frame, (300, 300))
	
	blob = cv2.dnn.blobFromImage(resized_image, (1/127.5), (300, 300), 127.5, swapRB=True)
	# print(blob.shape) # (1, 3, 300, 300)
	# pass the blob through the network and obtain the predictions and predictions
	net.setInput(blob) # net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
	# Predictions:
	predictions = net.forward()

	# loop over the predictions
	for i in np.arange(0, predictions.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		# predictions.shape[2] = 100 here
		confidence = predictions[0, 0, i, 2]
		# Filter out predictions lesser than the minimum confidence level
		# Here, we set the default confidence as 0.2. Anything lesser than 0.2 will be filtered
		if confidence > args["confidence"]:
			# extract the index of the class label from the 'predictions'
			# idx is the index of the class label
			# E.g. for person, idx = 15, for chair, idx = 9, etc.
			idx = int(predictions[0, 0, i, 1])
			# then compute the (x, y)-coordinates of the bounding box for the object
			box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
			# Example, box = [130.9669733   76.75442174 393.03834438 224.03566539]
			# Convert them to integers: 130 76 393 224
			(startX, startY, endX, endY) = box.astype("int")

			# Get the label with the confidence score
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			print("Object detected: ", label)
			# Draw a rectangle across the boundary of the object
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			# Put a text outside the rectangular detection
			# Choose the font of your choice: FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX, FONT_HERSHEY_COMPLEX, FONT_HERSHEY_SCRIPT_COMPLEX, FONT_ITALIC, etc.
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF

	# Press 'q' key to break the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer
fps.stop()

# Display FPS Information: Total Elapsed time and an approximate FPS over the entire video stream
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

# Destroy windows and cleanup
cv2.destroyAllWindows()
# Stop the video stream
vs.stop()

# In case you removed the opaque tape over your laptop cam, make sure you put them back on once finished ;)
# YAYYYYYYYYYY WE ARE DONE!

#Here’s a brief overview of what each function does:

#argparse: Parses command-line arguments.
#cv2.dnn.readNetFromCaffe(): Loads a Caffe model.#
#VideoStream(): Starts a video stream.
#time.sleep(): Pauses execution.
#FPS().start(): Starts FPS counter.
#cv2.resize(): Resizes images.
#net.setInput(): Sets the network input.
#net.forward(): Runs inference.
#cv2.rectangle(): Draws rectangles.
#cv2.putText(): Draws text.
#cv2.imshow(): Displays images.
#cv2.waitKey(): Waits for key events.
#cv2.destroyAllWindows(): Closes all OpenCV windows.
#vs.stop(): Stops the video stream.