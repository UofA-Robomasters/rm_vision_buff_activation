import cv2, sys, os, rospy, math
import numpy as np
from scipy.misc import imresize

root = os.path.dirname(os.path.abspath(__file__))
root = root+'/..'#'/number_searching'
sys.path.insert(0, root)
# print(root)
print(os.path.dirname(root))
from number_searching.grid_recognition import read_image_from_file

file_dir = None
is_debug_mode = True


"""
Major process of prompt lights searching
"""
def prompt_lights_searching(src_img):
    return src_img



"""
Main function (for testing)
"""
if __name__ == "__main__":
    """ ================ Testing with image files (START) ================ """
    # """
    # import .grid_recognition
    # from grid_recognition import read_image_from_file


    #load src image
    src_img = read_image_from_file()
    # src_img, number_boxes_regions_list, _ = number_search(src_img)
    src_img = prompt_lights_searching(src_img)

    cv2.imshow('src_img', src_img)
    key = cv2.waitKey(0)

    # """
    """ ================= Testing with image files (END) ================= """

    """ ================ Testing with video files (START) ================ """
    """
    # cam = cv2.VideoCapture('./../Buff2017.mp4')
    cam = cv2.VideoCapture('./../../buff_test_video_00.mpeg')

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = None#cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    frame_num = 1
    segment_num = 1
    frame_rate = 24
    recording = False

    while True:
        ret, frame = cam.read()
        assert ret == True

        # src_img, number_boxes_regions_list, _ = number_search(frame)

        cv2.imshow('src_img', src_img)

        key = cv2.waitKey(1000/frame_rate) & 0xff
        if key == ord('q'):
            break
    """
    """ ================= Testing with image files (END) ================= """

    cv2.destroyAllWindows()
