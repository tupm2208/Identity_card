import cv2 as cv
import numpy as np
import time
import shutil

positions = {
    'id': [(380 - 35, 155 - 38), (677 - 35, 187 - 38)],
    'name': [(340 - 35, 200 - 38), (723 - 35, 228 - 38)],
    'birth': [(377 - 35, 285 - 38), (723 - 35, 308 - 38)],
    'address1': [(415 - 35, 330 - 38), (723 - 35, 350 - 38)],
    'address2': [(274 - 35, 370 - 38), (723 - 35, 390 - 38)],
    'address3': [(495 - 35, 410 - 38), (723 - 35, 430 - 38)],
    'address4': [(274 - 35, 450 - 38), (723 - 35, 470 - 38)],
}

c_threshold = 0.5  # set threshold for bounding box values
nms = 0.4  # set threshold for non maximum supression
width = 416  # width of input image
height = 416  # height of input image

classesFile = "training/cfg/identity.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# initialize a list of colors to represent each possible class label
COLORS = np.random.randint(0, 255, size=(len(classes), 3),
                           dtype="uint8")

# PATH to weight and config files
config = 'training/cfg/yolov3-tiny.cfg'
weight = 'training/backup/yolov3-tiny_last.weights'

# Read the model using dnn
net = cv.dnn.readNetFromDarknet(config, weight)
cv.namedWindow('Image', cv.WINDOW_NORMAL)


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def handle_output(image, bboxes):
    bboxes = np.array(bboxes)
    min_x = np.min(bboxes[:, 0])
    min_y = np.min(bboxes[:, 1])
    max_x = np.max(bboxes[:, 0])
    max_y = np.max(bboxes[:, 1])

    bboxes[:, 0] -= min_x
    bboxes[:, 1] -= min_y

    img = image[min_y:max_y, min_x: max_x, :]
    bboxes = order_points(bboxes)
    a = bboxes[3].copy()
    bboxes[3] = bboxes[2]
    bboxes[2] = a

    pts1 = np.float32(bboxes)
    pts2 = np.float32([[0, 0], [730, 0], [0, 457], [730, 457]])
    print(pts1)
    print(pts2)
    matrix = cv.getPerspectiveTransform(pts1, pts2)

    result = cv.warpPerspective(img, matrix, (730, 457))

    cv.imshow('', result)
    cv.waitKey(0)


def detect(img_path):
    image = cv.imread(img_path)

    (H, W) = image.shape[:2]

    # Get the names of output layers
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # generate blob for image input to the network
    blob = cv.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()

    layersOutputs = net.forward(ln)

    end = time.time()

    print('runtime: ', 1 / (end - start))
    boxes = []
    confidences = []
    classIDs = []

    for output in layersOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > c_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Remove unnecessary boxes using non maximum suppression
    idxs = cv.dnn.NMSBoxes(boxes, confidences, c_threshold, nms)

    coordinates = []

    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])

            (w, h) = (boxes[i][2], boxes[i][3])
            cx = x + w//2
            cy = y + h//2
            coordinates.append((cx, cy))

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv.rectangle(image, (x, y), (x + w, y + h), color, 10)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 2)

    if len(coordinates) == 4:
        handle_output(image, coordinates)
    # else:
    #     shutil.move(img_path, img_path.replace('CMND', 'NO_CMND'))

    # show the output image
    cv.imshow("Image", image)
    cv.waitKey(0)


from glob import glob

image_list = glob('/home/tupm/SSD/datasets/identity_card/CMND/*')

# detect('/home/tupm/Pictures/cmt2.jpg')

for e in image_list:
    detect(e)
