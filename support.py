# Developed By Sashwat Jha ;)
# https://github.com/sashwatjha
import cv2
import numpy as np
from tensorflow.keras.models import load_model


def pre_process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)
    return img_threshold


def largest_contour(c1):
    points = np.array([])
    max_area = 0
    for i in c1:
        area = cv2.contourArea(i)
        if area > 50:
            l1 = cv2.arcLength(i, True)
            corner = cv2.approxPolyDP(i, 0.02 * l1, True)
            if area > max_area and len(corner) == 4:
                points = corner
                max_area = area
    return points, max_area


def rearrange(coordinates):
    coordinates = coordinates.reshape((4, 2))
    new = np.zeros((4, 1, 2), dtype=np.int32)
    temp = coordinates.sum(1)
    new[0] = coordinates[np.argmin(temp)]
    new[3] = coordinates[np.argmax(temp)]
    temp = np.diff(coordinates, axis=1)
    new[1] = coordinates[np.argmin(temp)]
    new[2] = coordinates[np.argmax(temp)]
    return new


# def noise_reduction(j):
#     thresh = cv2.threshold(j, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
#
#     cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     for c in cnts:
#         area = cv2.contourArea(c)
#         if area < 50:
#             cv2.drawContours(opening, [c], -1, 0, -1)
#
#     result = 255 - opening
#     result = cv2.GaussianBlur(result, (3, 3), 0)
#     cv2.waitKey()
#
#     return result


def split(img):
    r = np.vsplit(img, 9)
    all_sqrs = []
    for i in r:
        c = np.hsplit(i, 9)
        for j in c:
            # j = noise_reduction(j)
            all_sqrs.append(j)
    return all_sqrs


def predict(sqrs, ml_model):
    result = []
    for img in sqrs:
        img = np.asarray(img)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        predict1 = ml_model.predict(img)
        index = np.argmax(predict1, axis=-1)
        p_value = np.amax(predict1)
        if p_value > 0.45:
            result.append(index[0])
        else:
            result.append(0)
    return result


def num_list2img(img1, num_list, color=(0, 255, 0)):
    w = int(img1.shape[1] / 9)
    h = int(img1.shape[0] / 9)
    for i in range(0, 9):
        for j in range(0, 9):
            if num_list[(j * 9) + i] != 0:
                cv2.putText(img1, str(num_list[(j * 9) + i]),
                            (i * w + int(w / 2) - 10, int((j + 0.8) * h)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img1


def compare(img_array, scale):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        hor_con = [image_blank] * cols
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
            hor_con[x] = np.concatenate(img_array[x])
        output = np.vstack(hor)
    else:
        for x in range(0, rows):
            img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        output = hor
    return output
