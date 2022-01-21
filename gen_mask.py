from pathlib import Path
import time

import numpy as np
from PIL import Image
import cv2

directions = np.array([-1, 1])


def horizontal_mask(h, w, mask):
    # 先取一段横着的线
    hor_line_x1 = np.random.randint(0, w - 50)
    hor_line_x2 = hor_line_x1 + np.random.randint(0, 50)
    hor_line_width = max(hor_line_x2 - hor_line_x1, 30)
    ver_line_y1 = np.random.randint(0, h // 2)
    ver_line_y2 = np.random.randint(h // 2, h)
    direct = np.random.randint(0, 2)  # 表明左或者右
    idx = 0
    delta = 0
    for i in range(ver_line_y1, ver_line_y2):
        cur_hor_x1 = hor_line_x1 + delta
        # print(cur_hor_x1)
        cur_hor_line = [i for i in range(
            cur_hor_x1, min(255, cur_hor_x1 + hor_line_width))]
        mask[i, cur_hor_line] = 0
        if idx % 2 == 0:
            delta += directions[direct] * 1
        if idx % 6 == 0:
            # direct = 1 - direct
            direct = np.random.randint(0, 2)
            #print('now direct change:',direct)
            # delta = 0
        idx += 1


def vertical_mask(h, w, mask):
    # 先取一段横着的线
    ver_line_y1 = np.random.randint(0, h - 50)
    ver_line_y2 = ver_line_y1 + np.random.randint(0, 50)
    ver_line_width = max(ver_line_y2 - ver_line_y1, 30)
    hor_line_x1 = np.random.randint(0, w // 2)
    hor_line_x2 = np.random.randint(w // 2, w)
    direct = np.random.randint(0, 2)  # 表明左或者右
    idx = 0
    delta = 0
    for i in range(hor_line_x1, hor_line_x2):
        cur_ver_y1 = ver_line_y1 + delta
        cur_ver_line = [i for i in range(
            cur_ver_y1, min(255, cur_ver_y1 + ver_line_width))]
        # 这里的min是防止越界
        mask[cur_ver_line, i] = 0
        if idx % 2 == 0:
            delta += directions[direct] * 1
        if idx % 6 == 0:
            # direct = 1 - direct
            direct = np.random.randint(0, 2)
            # delta = 0
        idx += 1


def hor_ver_mask(h, w, mask, mode='hor'):  # 通用版，h和w反着传
    # 先取一段横着的线
    hor_line_x1 = np.random.randint(0, w - 20)
    hor_line_x2 = hor_line_x1 + np.random.randint(0, 20)
    hor_line_width = max(hor_line_x2 - hor_line_x1, 12)
    hor_line_width = min(hor_line_width, 16)
    ver_line_y1 = np.random.randint(0, h // 2)
    ver_line_y2 = np.random.randint(h // 2, h)
    ver_line_width = max(ver_line_y2 - ver_line_y1, 60)
    direct = np.random.randint(0, 2)  # 表明左或者右
    idx = 0
    delta = 0
    for i in range(ver_line_width):
        cur_hor_x1 = hor_line_x1 + delta
        # print(cur_hor_x1)
        cur_hor_line = [i for i in range(
            cur_hor_x1, min(h, cur_hor_x1 + hor_line_width))]
        if mode == 'hor':
            mask[i, cur_hor_line] = 0
        elif mode == 'ver':
            mask[cur_hor_line, i] = 0
        if idx % 2 == 0:
            delta += directions[direct] * 1
        if idx % 3 == 0:
            # direct = 1 - direct
            direct = np.random.randint(0, 2)
            #print('now direct change:',direct)
            # delta = 0
        idx += 1


def get_mask_ratio(mask):
    black = np.sum(mask == 0)
    ratio = black / (mask.shape[0] * mask.shape[1])
    return ratio


def rotate(mask):
    angle = np.random.randint(-180, 180)  # 角度也要随机
    m = cv2.getRotationMatrix2D(
        (mask.shape[1] / 2, mask.shape[0] / 2), angle, 1.0)
    mask = cv2.warpAffine(
        mask, m, (mask.shape[1], mask.shape[0]), borderValue=(255, 255, 255))
    return mask


def dilation(mask):  # 膨胀
    kernel_size = np.random.randint(2, 4)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel, iterations=4)
    return mask


def erode(mask):  # 腐蚀
    kernel_size = np.random.randint(2, 4)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.erode(mask, kernel, iterations=4)  # iterations是被递归的次数
    return mask


def add_holes(mask):
    h, w = np.shape(mask)
    black_white = np.array([0, 255])
    for i in range(h):
        for j in range(w):
            if mask[i][j] >= 119:
                continue
            new_pixel = np.random.choice(black_white, size=1, p=[0.8, 0.2])
            mask[i][j] = new_pixel
    return mask


def gen_mask(img_shape: tuple) -> np.ndarray:
    '''
    Generate a mask whose shape is `img_shape`.
    '''
    h, w = img_shape
    number_of_block = 6  # 遮挡条的个数
    mask = np.ones((h, w)) * 255
    for i in range(number_of_block):
        if i < number_of_block / 2:
            hor_ver_mask(h, w, mask)
        else:
            hor_ver_mask(w, h, mask, mode='ver')
    mask = rotate(mask)
    mask = dilation(mask)
    mask = erode(mask)
    mask = mask.astype('uint8')
    # mask = add_holes(mask)  # Add holes in mask, not useful for Oracle inpainting
    mask //= 255  # 先变成int，然后整除，就可以把不是黑的比如119.5这种灰灰的，变成黑的
    return mask
    

def gen_masks(img_shape: tuple, cnt: int) -> None:
    '''
    Generate `cnt` masks of shape `img_shape`.
    '''
    mask_ratio = []
    for idx in range(cnt):
        mask = gen_mask(img_shape)
        # 之后因为ToTensor()那里会除以255，所以如果是全0和1的话，就会变成0.0039，到时候y_comp就会出问题
        # 所以必须在这里变成255的先
        # 因为y_comp = mask * y_true + (1 - mask) * y_pred，mask是0.0039，所以y_pred就会
        # 占很大的成分，导致y_comp看起来跟y_pred一样
        img = Image.fromarray(mask * 255).convert('1')
        out_file = Path('data', 'mask', f'{idx}.png')
        img.save(out_file)

        # Compute mask ratio
        mask_ratio.append(get_mask_ratio(mask))
    
    avg_mask_ratio = sum(mask_ratio) / len(mask_ratio)
    print('Average mask ratio:', avg_mask_ratio)


def main():
    np.random.seed(0)
    img_shape = (128, 128)
    cnt = 1000
    gen_masks(img_shape, cnt)


if __name__ == '__main__':
    main()
