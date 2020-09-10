import numpy as np
import cv2
import random
import json
from PIL import Image, ImageFont, ImageDraw
from utils import _generate_horizontal_text, resize, load_address
from glob import glob

positions = {
    'id': [(380, 155), (677, 187)],
    'name': [(340, 200), (723, 228)],
    'birth': [(377, 285), (723, 308)],
    'address1': [(415, 330), (723, 350)],
    'address2': [(274, 370), (723, 390)],
    'address3': [(495, 410), (723, 430)],
    'address4': [(274, 450), (723, 470)],
}

address_list = load_address()
pose_list = glob('assets/anh_the/*.jpg')


def get_random_info():
    id_str = ''.join([str(random.randint(0, 9)) for _ in range(9)])
    name = 'PHẠM MINH TƯ'
    birth = ''.join([str(random.randint(0, 9)) for _ in range(2)]) + '-' + \
            ''.join([str(random.randint(0, 9)) for _ in range(2)]) + '-' + \
            ''.join([str(random.randint(0, 9)) for _ in range(4)])

    text = random.choice(address_list).strip()
    address1, address2 = text.split('\t')

    text = random.choice(address_list).strip()
    address3, address4 = text.split('\t')

    return [
        {'kind': 'id', 'text': id_str, 'space_width': 3, 'height': 25},
        {'kind': 'name', 'text': name, 'space_width': 1, 'height': 26},
        {'kind': 'birth', 'text': birth, 'space_width': 1, 'height': 18},
        {'kind': 'address1', 'text': address1, 'space_width': 1, 'height': 23 if 'g' in address1 or 'y' in address1 else 20},
        {'kind': 'address2', 'text': address2, 'space_width': 1, 'height': 23 if 'g' in address2 or 'y' in address2 else 20},
        {'kind': 'address3', 'text': address3, 'space_width': 1, 'height': 23 if 'g' in address3 or 'y' in address3 else 20},
        {'kind': 'address4', 'text': address4, 'space_width': 1, 'height': 23 if 'g' in address4 or 'y' in address4 else 20}
    ]


def paste_image(background_image, text, kind, space_width=3, height=25):
    image, mask = _generate_horizontal_text(text, font_size=32, space_width=space_width)
    image = resize(image, height=height)

    idw = image.size[0]
    x = (positions[kind][1][0] + positions[kind][0][0] - idw) // 2 + random.randint(-10, 10)
    y = positions[kind][0][1]

    background_image.paste(image, (x, y), image)

    return image


def paste_pose(background):
    # (75, 225) (243, 444)
    img_path = random.choice(pose_list)

    image = Image.open(img_path)

    image = resize(image, 166, 219)
    background.paste(image, (75, 225))



def get_card():
    path = '/home/tupm/projects/Identity_card/assets/background/cmt_bacground.png'

    image = Image.open(path)

    info = get_random_info()

    for e in info:
        kind = e['kind']
        text = e['text']
        space_width = e['space_width']
        height = e['height']

        paste_image(image, text, kind, space_width, height)

    paste_pose(image)

    image.show()


if __name__ == '__main__':
    get_card()
    # cv2.imshow('', cv2.imread('/home/tupm/projects/Identity_card/assets/background/cmt_bacground.png'))
    # cv2.waitKey(0)
    # read_location()
