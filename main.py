# Developed By Sashwat Jha ;)
# https://github.com/sashwatjha
print("Running....")
# used for suppressing terminal warnings------------------------------
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# --------------------------------------------------------------------

import solver
from support import *

path = "Resources/1.jpg"
height_img = 450
width_img = 450
ml_model = load_model('DigitDetectionModel_V2.h5')

img = cv2.imread(path)
img = cv2.resize(img, (width_img, height_img))
blank = np.zeros((height_img, width_img, 3), np.uint8)
threshold = pre_process(img)

contour = img.copy()
lg_contour = img.copy()
c, h = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

large, lg_area = largest_contour(c)

if large.size != 0:
    large = rearrange(large)
    cv2.drawContours(lg_contour, large, -1, (0, 0, 255), 30)
    x1 = np.float32(large)
    x2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
    t_matrix = cv2.getPerspectiveTransform(x1, x2)

    img_warped = cv2.warpPerspective(img, t_matrix, (width_img, height_img))

    img_warped = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

    sqrs = split(img_warped)

    num_list = predict(sqrs, ml_model)

    num_list = np.asarray(num_list)
    position_list = np.where(num_list > 0, 0, 1)

    sudoku = np.array_split(num_list, 9)
    print(sudoku)
    try:
        solver.solve(sudoku)
    except:
        pass

    solved_list = []
    for row in sudoku:
        for i in row:
            solved_list.append(i)

    solved_digit = blank.copy()
    s_numbers = solved_list * position_list
    new_digits_img = num_list2img(solved_digit, s_numbers)

    y2 = np.float32(large)
    y1 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
    t_matrix = cv2.getPerspectiveTransform(y1, y2)
    new_img = img.copy()
    new_img = cv2.warpPerspective(new_digits_img, t_matrix, (width_img, height_img))
    new_img = cv2.addWeighted(new_img, 1, img, 0.5, 1)

image_array = ([img, new_img])
stacked = compare(image_array, 1)
cv2.imshow('Result', stacked)
print("Ended....")
cv2.waitKey(0)
