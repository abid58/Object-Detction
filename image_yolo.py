""" This file uses the YOLO template to build vg-NMS on images.
    @author: Abid Hossain
"""
import numpy as np
import argparse
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-y", "--path", required=True, help="Path of YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5)
ap.add_argument("-t", "--threshold", type=float, default=0.3)
args = vars(ap.parse_args())

# COCO class labels
classLabelPath = os.path.sep.join([args["path"], "coco.names"])
labels = open(classLabelPath).read().strip().split("\n")
np.random.seed(2)


# load weights and confiuguration of the pretrained model
labelcolors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
weightsPath = os.path.sep.join([args["path"], "yolov3.weights"])
configPath = os.path.sep.join([args["path"], "yolov3.cfg"])


# derive the paths to the YOLO weights and model configuration
print("starting model.. ")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
image = cv2.imread(args["input"])
(image_H, image_W) = image.shape[:2]


#retrieve output
layers = net.getLayerNames()
lastlayer = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
cherypicked_blob = cv2.dnn.blobFromImage(image, 1.0/256.0, (416, 416), swapRB=True, crop=False)
net.setInput(cherypicked_blob)


#apply pixel-based object detector
start_time = time.time()
outputs = net.forward(lastlayer)
end_time = time.time()
print("Time taken by YOLO is {:.6f} seconds".format(end_time - start_time))

# output:bounding boxes, associated probabilities

""" Pixel-based """
confidences = []
classids = []
boundaryboxes = []


for output in outputs:
	for detection in output:

		scores = detection[5:]
		classid = np.argmax(scores) # take the max prob object
		confidence = scores[classid]

        # proceed only is confidence is more than predefined threshold
		if confidence > args["confidence"]:
			box = detection[0:4] * np.array([image_W, image_H, image_W, image_H])
			(boxcenterX, boxcenterY, boxwidth, boxheight) = box.astype("int")

            # find topleft corner
			x = int(boxcenterX - (boxwidth / 2))
			y = int(boxcenterY - (boxheight / 2))

			boundaryboxes.append([x, y, int(boxwidth), int(boxheight)])
			confidences.append(float(confidence))
			classids.append(classid)

# apply NMS and draw the boxes
idxs1 = cv2.dnn.NMSBoxes(boundaryboxes, confidences, args["confidence"], args["threshold"], 1.0)



"""
# amodal

confidences = []
classids = []
boundaryboxes = []

# Adaptive thresholding vg-NMS
for output in outputs:
	for detection in output:
		scores = detection[5:]
		classid = np.argmax(scores) # take the max prob object
		confidence = scores[classid]
        # proceed only is confidence is more than predefined threshold
		if confidence > args["confidence"]:
			box = detection[0:4] * np.array([image_W, image_H, image_W, image_H])
			(boxcenterX, boxcenterY, boxwidth, boxheight) = box.astype("int")
            # find topleft corner
			x = int(boxcenterX - (boxwidth / 2))
			y = int(boxcenterY - (boxheight / 2))
			boundaryboxes.append([x, y, int(boxwidth), int(boxheight)])
			confidences.append(float(confidence))
			classids.append(classid)

# apply NMS and draw the boxes
idxs2 = cv2.dnn.NMSBoxes(boundaryboxes, confidences, args["confidence"], args["threshold"], 0.9) # eta = 0.9
"""





if len(idxs1) > 0:
	for i in idxs1.flatten():
		(x, y) = (boundaryboxes[i][0], boundaryboxes[i][1])
		(w, h) = (boundaryboxes[i][2], boundaryboxes[i][3])
		box_color = [int(color) for color in labelcolors[classids[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)
		text = "{}: {:.4f}".format(labels[classids[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)


# show the output imagege
cv2.imwrite(args["output"], image)
cv2.imshow("output image", image)
cv2.waitKey(0)
