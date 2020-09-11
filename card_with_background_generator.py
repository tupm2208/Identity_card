import random
import cv2
import numpy as np
import os
from PIL import Image
from card_generator import get_card
from glob import glob
from utils import resize
from multiprocessing import Pool
from tqdm import tqdm

background_path_list = glob('assets/background2/*')




def after_resize(bg_size, card, coordinates):
    min_size=100
    left, top, right, bottom = card.getbbox()
    bw, bh = bg_size

    if bw < right*bh/bottom:
        new_size = random.randint(min_size, bw)
        card = resize(card, width=new_size)
    else:
        new_size = random.randint(min_size, bh)
        card = resize(card, height=new_size)

    nw, nh = card.size

    coordinates = coordinates.astype('float')


    coordinates[:, 0] *= nw/right
    coordinates[:, 1] *= nh/bottom

    return card, coordinates


def get_output_file(idx):
    global output_folder
    folder = str(idx//1000)
    output_img_path = os.path.join(output_folder, folder, f'{idx}.jpg')
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)

    return output_img_path


def get_label_yolo(image_path, coordinates, img_size):
    w, h = img_size
    padding_x = w//20
    padding_y = h//20

    coordinates[:, 0] /= w
    coordinates[:, 1] /= h

    iw = 2*padding_x/w
    ih = 2*padding_y/h
    label_path = image_path.replace('.jpg', '.txt')
    with open(label_path, 'w+') as ft:
        for idx, coor in enumerate(coordinates):
            x, y = coor
            ft.write(f'{idx} {x} {y} {iw} {ih}\n')


mask_list = glob('assets/mask_brightness/*')


def add_shadow(image):
    ## Conversion to HLS
    mask_path = random.choice(mask_list)
    mask = cv2.imread(mask_path, 0)
    image = np.array(image)
    image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(image_HSV)
    height, width, _ = image.shape
    mask = cv2.resize(mask, (width, height))
    ratio = random.uniform(0.5, 1.5)
    points = np.argwhere(mask > 0)
    for point in points:
        temp = v[point[0], point[1]] * ratio
        if temp > 255:
            v[point[0], point[1]] = 255
        else:
            v[point[0], point[1]] = temp
    v = v.astype('uint8')
    ## Conversion to RGB
    final_hsv = cv2.merge((h, s, v))
    final_hsv = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    final_hsv = Image.fromarray(final_hsv)
    return final_hsv

def gen_card_on_background(idx):

    background_path = random.choice(background_path_list)
    background = Image.open(background_path)
    background = resize(background, width=512)
    bg_size = background.size
    card, coordinates = get_card()
    card, coordinates = after_resize(bg_size, card, coordinates)
    px, py = np.array(bg_size) - np.array(card.size)

    px = random.randint(0, px)
    py = random.randint(0, py)
    background.paste(card, (px, py), card)
    output_img_path = get_output_file(idx)
    
    background = add_shadow(background)

    background.save(output_img_path, 'JPEG')

    coordinates[:, 0] += px
    coordinates[:, 1] += py
    

    return output_img_path, coordinates, bg_size



if __name__ == '__main__':
    output_folder = "train"
    f = open(os.path.join(output_folder, 'labels.txt'), 'w+')
    
    pool = Pool(4)

    for output_img_path, coordinates, img_size in tqdm(pool.imap_unordered(gen_card_on_background, range(10))):
        get_label_yolo(output_img_path, coordinates, img_size)
        f.write(f'{output_img_path}\n')

    f.close()
