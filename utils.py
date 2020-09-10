import json
from PIL import Image, ImageFont, ImageDraw


def _generate_horizontal_text(text, font='assets/fonts/pala.ttf', font_size=16, space_width=1, character_spacing=0,
                              fit=True, word_split=' '):

    image_font = ImageFont.truetype(font, font_size)

    space_width = int(image_font.getsize(" ")[0] * space_width)

    if word_split:
        splitted_text = []
        for w in text.split(" "):
            splitted_text.append(w)
            splitted_text.append(" ")
        splitted_text.pop()
    else:
        splitted_text = text

    piece_widths = [
        image_font.getsize(p)[0] if p != " " else space_width for p in splitted_text
    ]
    text_width = sum(piece_widths)
    if not word_split:
        text_width += character_spacing * (len(text) - 1)

    text_height = max([image_font.getsize(p)[1] for p in splitted_text])

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")
    txt_mask_draw.fontmode = "1"

    fill = (0, 0, 0)

    for i, p in enumerate(splitted_text):
        txt_img_draw.text(
            (sum(piece_widths[0:i]) + i * character_spacing * int(not word_split), 0),
            p,
            fill=fill,
            font=image_font,
        )
        txt_mask_draw.text(
            (sum(piece_widths[0:i]) + i * character_spacing * int(not word_split), 0),
            p,
            fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
            font=image_font,
        )

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox())
    else:
        return txt_img, txt_mask


def resize(image, width=None, height=None):
    w, h = image.size

    if not width and not height:
        return image

    if width and height:
        return image.resize((width, height))

    if width:
        height = int(width * (h/w))
    else:
        width = int(height * (w/h))

    return image.resize((width, height))


def load_address():
    with open('assets/local.txt', encoding='utf8') as f:
        return f.readlines()

def convert_data():
    with open('assets/local.json', encoding='utf8') as f:
        data = json.load(f)

    f = open('assets/local.txt', 'w+', encoding='utf8')

    # print(data[0]['districts'][0]['wards'])
    for tp in data:
        tp_name = tp['name']
        for district in tp['districts']:
            district_name = district['name']
            for ward in district['wards']:
                ward_name = ward['name']
                if ward_name.isnumeric():
                    ward_name = 'Phường ' + ward_name

                f.write(f'{ward_name}\t{district_name}, {tp_name}\n')