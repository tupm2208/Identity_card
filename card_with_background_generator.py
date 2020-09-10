import random
import cv2
from PIL import Image
from card_generator import get_card
from glob import glob
from utils import resize

background_path_list = glob('assets/background2/*')


def gen_card_on_background():

    background_path = random.choice(background_path_list)
    # cv2.imshow('', cv2.imread(background_path))
    # cv2.waitKey(0)
    background = Image.open(background_path)
    background = resize(background, width=1024)
    card, coordinates = get_card()
    background.paste(card, (0, 0), card)
    background.show()


if __name__ == '__main__':
    gen_card_on_background()