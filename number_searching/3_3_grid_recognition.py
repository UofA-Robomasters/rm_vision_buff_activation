import cv2
import numpy as np
import sys
import rospy

file_dir = None

"""
Parse the file direction from argv and load the image
"""
def read_image_from_file():

    # parse the file direction
    file_dir = None
    if(len(sys.argv)>1):
        file_dir = sys.argv[1]
    else:
        print("Error: No file name provided")
        exit()

    # load the image and return
    img = cv2.imread(file_dir)
    return img



"""
Pre-processing image
"""
def preprocessing_for_number_searching(src_img):
    # convert source iamge to gray scale
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    # threshold
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                 cv2.THRESH_BINARY, 15, 3)
    # ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # blur
    gray = cv2.medianBlur(gray,3)

    return gray



"""
Analysis and filter contours
"""
def analysis_and_filter_contours_for_number_searching(contours):
    ratio = 28.0 / 16.0
    sudokuWidth = 127
    sudokuHeight = 71
    angleTolerance = 6
    ratioToleranceRate = 0.2
    dimensionsToleranceRate = 0.4

    num = 0

    output = list()
    for contour in contours:
        tempRect = cv2.minAreaRect(contour)
        width = tempRect[1][0]
        height = tempRect[1][1]

        if not (width > height):
            # tempRect = cv2.boxPoints((tempRect[0],(tempRect[1][0],tempRect[1][1]),tempRect[2] + 90.0))
            tempRect = (tempRect[0],(tempRect[1][0],tempRect[1][1]),tempRect[2] + 90.0)
            width = tempRect[1][0]
            height = tempRect[1][1]

        if(height==0):
            height = 1

        # if (num < 20):
        #     # print(contour)
        #     print(tempRect)
        #     print("W:", width, "H:", height)
        # num += 1
        # print(num)

        ratio_cur = width / height

        if (ratio_cur > (1.0-ratioToleranceRate) * ratio and \
            ratio_cur < (1.0+ratioToleranceRate) * ratio and \
			width > (1.0-dimensionsToleranceRate) * sudokuWidth and \
            width < (1.0+dimensionsToleranceRate) * sudokuWidth and \
			height > (1.0-dimensionsToleranceRate) * sudokuHeight and \
            height < (1.0+dimensionsToleranceRate) * sudokuHeight and \
			((tempRect[2] > -angleTolerance and tempRect[2] < angleTolerance) or \
              tempRect[2] < (-180+angleTolerance) or \
              tempRect[2] > (180-angleTolerance))
        ):
              output.append(contour)

    return output



"""
Main function
"""
if __name__ == "__main__":
    #load src image
    src_img = read_image_from_file()

    #preprocessing image
    processed_img = preprocessing_for_number_searching(src_img)
    # cv2.imshow('processed_img', processed_img)
    # cv2.waitKey(0)

    #find contours
    im2, contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(src_img, contours, -1, (0,255,0), 3)
    # print(contours[0])

    #Analysis
    contours = analysis_and_filter_contours_for_number_searching(contours)
    cv2.drawContours(src_img, contours, -1, (0,255,0), 3)
    print("The size of contour list is:", len(contours))
    print(cv2.minAreaRect(contours[0]))

    cv2.imshow('src_img', src_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
