"""
    This file uses the YOLO template to build vg-NMS on videos.
    @author: Abid Hossain
"""
import numpy as np
import argparse
import imutils
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-y", "--path", required=True, help="path of YOLO directory")
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


print("starting model.. ")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layers = net.getLayerNames()
layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]


#retrieve input
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
#retrieve frames in video
prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
    else cv2.CAP_PROP_FRAME_COUNT
total = int(vs.get(prop))
print("Total frames in video : {}".format(total))


#for each frame
while True:
	#retrieve frame
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    cherypicked_blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(cherypicked_blob)
    # apply pixel-based object detector
    start_time = time.time()
    layeroutputs = net.forward(layers)
    end_time = time.time()
    #setup bounding boxes & confidences per class
    boundaryboxes = []
    confidences = []
    classids = []
    #NMS runs on pixel-based detector
    for output in layeroutputs:
        for detection in output:
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (boxcenterX, boxcenterY, boxwidth, boxheight) = box.astype("int")
                x = int(boxcenterX - (boxwidth / 2))
                y = int(boxcenterY - (boxheight / 2))
                boundaryboxes.append([x, y, int(boxwidth), int(boxheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    # pixel based
    idxs1 = cv2.dnn.NMSBoxes(boundaryboxes, confidences, args["confidence"], args["threshold"])

    """
    # amodal
    # Adaptive thresholding
    idxs2 = cv2.dnn.NMSBoxes(boundaryboxes, confidences, args["confidence"], args["threshold"], 0.9) # eta = 0.9
    """

    #draw detections
    if len(idxs1) > 0:
        for i in idxs1.flatten():
            (boxx, boxy) = (boundaryboxes[i][0], boundaryboxes[i][1])
            (boxw, boxh) = (boundaryboxes[i][2], boundaryboxes[i][3])
            color = [int(c) for c in labelcolors[classids[i]]]
            cv2.rectangle(frame, (boxx, boxy), (boxx + boxw, boxy + boxh), color, 2)
            text = "{}: {:.4f}".format(labels[classids[i]], confidences[i])
            cv2.putText(frame, text, (boxx, boxy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
        #timing analysis
        if total > 0:
            print("single frame took {:.4f}".format((end_time - start_time)))
            print("estimated total time to finish {:.4f}".format((end_time - start_time) * total))
    writer.write(frame)

print('Done')
writer.release()
vs.release()





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
