import cv2
import os
from matplotlib import pyplot as plt
from gluoncv import utils
import numpy as np


target_h = 700
target_w = 700
orig_x_bbox_min = 885
orig_y_bbox_min = 213
obj_height = 515
obj_width = 316


def crop_image(img, target_h, target_w, bbox_x_min, bbox_y_min, obj_height, obj_width):

    orig_img = cv2.imread(img, 1)

    # Height = Y axis | Width = X axis
    cur_h, cur_w, _ = orig_img.shape
    y_ax_crop = cur_h - target_h
    x_ax_crop = cur_w - target_w

    new_x_bbox_min = orig_x_bbox_min - x_ax_crop/2
    new_y_bbox_min = orig_y_bbox_min - y_ax_crop/2

    new_x_min = int(x_ax_crop/2)
    new_x_max = int(cur_w - (x_ax_crop/2))
    new_y_min = int(y_ax_crop/2)
    new_y_max = int(cur_h - (y_ax_crop/2))



    new_bbox = [new_x_bbox_min, new_y_bbox_min, new_x_bbox_min+obj_width, new_y_bbox_min+obj_height]
    #boxes = np.array([new_bbox])
    #ids = np.array([0])
    #class_names = ['skidsteer']

    cropped = orig_img[new_y_min:new_y_max, new_x_min:new_x_max]

    #ax = utils.viz.plot_bbox(cropped, boxes, labels=ids, class_names=class_names)
    #plt.show()

    return cropped, new_bbox

