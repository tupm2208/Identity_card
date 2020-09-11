import random
import numpy as np
import cv2
from PIL import Image
from utils import _generate_horizontal_text, resize, load_address
from glob import glob

positions = {
    'id': [(380-35, 155-38), (677-35, 187-38)],
    'name': [(340-35, 200-38), (723-35, 228-38)],
    'birth': [(377-35, 285-38), (723-35, 308-38)],
    'address1': [(415-35, 330-38), (723-35, 350-38)],
    'address2': [(274-35, 370-38), (723-35, 390-38)],
    'address3': [(495-35, 410-38), (723-35, 430-38)],
    'address4': [(274-35, 450-38), (723-35, 470-38)],
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
    background.paste(image, (75-35, 225-38))


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def affine_transform(image):
    width, height = image.size

    corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])
    m = random.randint(-8, 8) * 0.1
    xshift = abs(m) * height
    new_width = width + int(round(xshift))
    image = image.transform((new_width, height), Image.AFFINE, (1, m, -xshift if m > 0 else 0, 0, 1, 0))
    cornersx = corners[:, 0] + np.round((height - corners[:, 1]) * m + (xshift if m < 0 else 0))
    width, height = image.size
    m = random.randint(-8, 8) * 0.1
    yshift = abs(m) * width
    new_height = height + int(round(yshift))
    image = image.transform((width, new_height), Image.AFFINE, (1, 0, 0, m, 1, -yshift if m > 0 else 0), Image.BICUBIC)
    cornersy = corners[:, 1] + np.round((width - cornersx) * m + (yshift if m < 0 else 0))

    return image, np.array(list(zip(cornersx, cornersy)), dtype='int')


def get_card():
    path = 'assets/background/cmt_bacground.png'

    image = Image.open(path).convert('RGBA')

    info = get_random_info()

    for e in info:
        kind = e['kind']
        text = e['text']
        space_width = e['space_width']
        height = e['height']

        paste_image(image, text, kind, space_width, height)

    paste_pose(image)

    image, coordinates = affine_transform(image)

    bbox = image.getbbox()

    coordinates[:, 1] -= bbox[1]

    image = image.crop(bbox)

    return image, coordinates


if __name__ == '__main__':
    get_card().show()
    # cv2.imshow('', cv2.imread('assets/background/cmt_bacground.png'))
    # cv2.waitKey(0)
    # read_location()
