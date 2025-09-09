import cv2
import numpy as np
from collections import OrderedDict
import math

class CentroidTracker:
    def __init__(self, maxDisappeared=40, maxDistance=80, history_len=30):
        self.nextObjectID = 0
        self.objects = OrderedDict()       
        self.boxes = OrderedDict()         
        self.disappeared = OrderedDict()   
        self.counted = OrderedDict()       
        self.centroid_history = OrderedDict()  
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.history_len = history_len

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = centroid
        self.boxes[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        self.counted[self.nextObjectID] = False
        self.centroid_history[self.nextObjectID] = [centroid]
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.boxes[objectID]
        del self.disappeared[objectID]
        del self.counted[objectID]
        del self.centroid_history[objectID]

    def update(self, rects):
        # rects: list of (startX, startY, endX, endY)
        if len(rects) == 0:
            # mark all existing objects as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, self.boxes

        # compute input centroids
        inputCentroids = []
        for (startX, startY, endX, endY) in rects:
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids.append((cX, cY))

        # if no objects yet -> register all inputs
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
        else:
            # build distance matrix between existing object centroids and input centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = []
            for oc in objectCentroids:
                row = []
                for ic in inputCentroids:
                    d = math.hypot(oc[0] - ic[0], oc[1] - ic[1])
                    row.append(d)
                D.append(row)

            # create a list of (dist, rowIdx, colIdx) and sort
            entries = []
            for i in range(len(D)):
                for j in range(len(D[0])):
                    entries.append((D[i][j], i, j))
            entries.sort(key=lambda x: x[0])

            usedRows = set()
            usedCols = set()

            # greedy match by increasing distance
            for dist, row, col in entries:
                if row in usedRows or col in usedCols:
                    continue
                if dist > self.maxDistance:
                    continue
                objectID = objectIDs[row]
                # update object data
                self.objects[objectID] = inputCentroids[col]
                self.boxes[objectID] = rects[col]
                self.disappeared[objectID] = 0
                # append centroid history (limit length)
                self.centroid_history[objectID].append(inputCentroids[col])
                if len(self.centroid_history[objectID]) > self.history_len:
                    self.centroid_history[objectID].pop(0)

                usedRows.add(row)
                usedCols.add(col)

            # any unmatched existing object -> increment disappeared
            for row in range(len(objectIDs)):
                if row not in usedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # any unmatched input centroid -> register as new object
            for col in range(len(inputCentroids)):
                if col not in usedCols:
                    self.register(inputCentroids[col], rects[col])

        return self.objects, self.boxes

# -------------------- main counting script --------------------
VIDEO_PATH = "./video1.mp4"
FRAME_WIDTH = 900   # choose to fit your screen
FRAME_HEIGHT = 600
MIN_AREA = 1500     # tune this: bigger for fewer false small blobs
MAX_DISTANCE = 80   # max distance for centroid matching (tune)
MAX_DISAPPEARED = 30

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error opening video:", VIDEO_PATH)
    exit()

tracker = CentroidTracker(maxDisappeared=MAX_DISAPPEARED, maxDistance=MAX_DISTANCE)
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=True)

total_count = 0
line_y = FRAME_HEIGHT // 2
cv2.namedWindow("Traffic Video", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    blur = cv2.GaussianBlur(frame, (5,5), 0)

    # background subtraction
    fgmask = fgbg.apply(blur)

    # threshold to remove shadows (grayish) and noise
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    # morph ops to remove noise and close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # basic size filter: skip tiny boxes
        if w < 30 or h < 30:
            continue
        rects.append((x, y, x + w, y + h))

    objects, boxes = tracker.update(rects)

    # draw counting line
    cv2.line(frame, (0, line_y), (FRAME_WIDTH, line_y), (255, 0, 0), 2)

    # examine tracked objects for counting
    for objectID, centroid in objects.items():
        box = boxes.get(objectID, None)
        if box is not None:
            (startX, startY, endX, endY) = box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {objectID}", (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

        # counting logic (top -> bottom). Use centroid history to avoid jitter
        if not tracker.counted[objectID]:
            hist = tracker.centroid_history[objectID]
            if len(hist) >= 2:
                prev_y = hist[-2][1]
                curr_y = hist[-1][1]
                # vehicle moved from above line to below line
                if prev_y < line_y and curr_y >= line_y:
                    total_count += 1
                    tracker.counted[objectID] = True

    # display counter
    cv2.putText(frame, f"Count: {total_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # optional: show mask in small window for debugging
    cv2.imshow("Mask", cv2.resize(fgmask, (FRAME_WIDTH//3, FRAME_HEIGHT//3)))
    cv2.imshow("Traffic Video", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

