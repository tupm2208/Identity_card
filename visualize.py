import cv2
import numpy as np

def get_original_coordinate(line, shape):
    w, h = shape

    data = np.array(line.split(' '), dtype=float)[1:]

    data[2] *= w
    data[3] *= h

    data[0] *= w
    
    data[1] *= h
    

    cx = int(data[0])
    cy = int(data[1])

    x1 = cx - int(data[2]/2)
    y1 = cy - int(data[3]/2)

    x2 = cx + int(data[2]/2)
    y2 = cy + int(data[3]/2)

    return cx, cy, x1, y1, x2, y2


def show_image(image_path):

    image = cv2.imread(image_path)
    h, w, _ = image.shape
    with open(image_path.replace('.jpg', '.txt')) as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()
            cx, cy, x1, y1, x2, y2 = get_original_coordinate(line, (w, h))

            # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(image, (cx, cy), 3, (255, 0, 0), 2)
    
    cv2.imshow('', image)
    cv2.waitKey(0)


show_image('/home/tupm/projects/Identity_card/train/0/2.jpg')