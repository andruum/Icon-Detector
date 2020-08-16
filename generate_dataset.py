import os
import random

import cv2
import numpy as np
import pandas as pd
import glob


def put_icon(img, icon, bbox_point, size):
    img_width = img.shape[1]
    img_height = img.shape[0]

    x,y = bbox_point

    if x >= img_width or y >= img_height:
        return img

    icon = cv2.resize(icon, (size,size), interpolation=cv2.INTER_NEAREST)
    h, w = size, size

    if x + w > img_width:
        w = img_width - x
        icon = icon[:, :w]

    if y + h > img_height:
        h = img_height - y
        icon = icon[:h]

    overlay_image = icon[..., :3]
    mask = icon[..., 3:] / 255.0

    img[y:y + h, x:x + w] = (1.0 - mask) * img[y:y + h, x:x + w] + mask * overlay_image

    return img


def load(path, color=cv2.COLOR_BGR2RGB):
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), color)


def get_pos(area, icon_size, section):
    section %= 4
    w = (area[2]-area[0])//2
    h = (area[3]-area[1])//2

    x = random.randint(area[0] + w*(section%2) , area[2] - icon_size - w*(1-section%2))
    y = random.randint(area[1] + h*(section//2), area[3] - icon_size - h*(1-section//2))

    return x, y


def main():

    backgrounds = [
        {'img': load("data/backgrounds/0.jpg"), 'area': [31, 35, 255, 531], 'icon_sizes': (30, 50)},
        {'img': load("data/backgrounds/1.jpg"), 'area': [41, 88, 316, 547], 'icon_sizes': (30, 60)} ,
        {'img': load("data/backgrounds/2.jpg"), 'area': [19, 66, 159, 311], 'icon_sizes': (20, 30)},
        {'img': load("data/backgrounds/3.jpg"), 'area': [53, 98, 585, 1205], 'icon_sizes': (60, 90)},
    ]

    dataset_size = 150
    validation_size = int(dataset_size*0.1)
    icons_per_image = 4

    class_output = 'data/icons/classes/images'
    csv_output = 'data/icons/classes/'
    src_output = 'data/icons/src'
    icons_source = "data/icons_samsung/*.png"

    os.makedirs(class_output, exist_ok=True)
    os.makedirs(csv_output, exist_ok=True)
    os.makedirs(src_output, exist_ok=True)

    icons = []

    for icon_p in glob.glob(icons_source):
        icon = load(icon_p, color=cv2.COLOR_BGRA2RGBA)
        white = np.ones((*icon.shape[:2],3))*255
        icon_to_save = put_icon(white, icon, (0,0), icon.shape[0])
        icon_to_save = np.float32(icon_to_save)
        icon_to_save = cv2.cvtColor(icon_to_save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(class_output,f"{len(icons)}.jpg"), icon_to_save)
        icons.append(icon)

    random.shuffle(icons)

    data = pd.DataFrame()

    for d in range(dataset_size):
        bg = backgrounds[d % len(backgrounds)]
        bg_image = bg['img'].copy()
        icon_sizes = bg['icon_sizes']
        for i in range(icons_per_image):
            gt_id = d * icons_per_image + i
            icon_id = gt_id % len(icons)
            icon = icons[icon_id]
            icon_size = random.randint(icon_sizes[0],icon_sizes[1])

            x,y = get_pos(bg['area'], icon_size, i)

            put_icon(bg_image, icon, (x,y), icon_size)

            obj = {
                'gtbboxid': gt_id,
                'classid': icon_id,
                'imageid': d,
                'lx': x / bg_image.shape[1],
                'rx': (x+icon_size) / bg_image.shape[1],
                'ty': y / bg_image.shape[0],
                'by': (y+icon_size) / bg_image.shape[0],
                'difficult': 0,
                'split': "val-new-cl" if d > dataset_size-validation_size else "train"
            }

            df = pd.DataFrame(obj, index=[gt_id])
            df = df.set_index('gtbboxid')
            data = data.append(df)

        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(src_output,f"{d}.jpg"), bg_image)

    data.to_csv(os.path.join(csv_output, 'data.csv'))

if __name__ == '__main__':
    main()
