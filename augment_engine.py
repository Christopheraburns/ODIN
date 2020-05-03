# augment_engine takes the images rendered by Blender and buids a dataset by resizing, moving and appending the image
# to random backgrounds
# burnsca@

import cv2
import numpy as np
from numpy import linspace
from math import sin, cos
import os
import math
import boto3
import random
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import logging
import time
import shutil
import datetime
import backgrounds
import sys

logging.basicConfig(filename='runtime.log', level=logging.INFO)

s3_background_bucket = 'odin-bck'
background_generator = None
m_length = 0  # Num of entries in background (minus 1 for zero based)
max_base = 600

# S3 Folder structure
#s3_output = s3_bucket + '/output/' + job_id + '/VOC'
#s3_train_image_output = s3_output + "/VOCTrain/JPEGImages/"
#s3_train_annot_output = s3_output + "/VOCTrain/Annotations/"
#s3_train_imageset_output = s3_output + "/VOCTrain/ImageSets/Main/"
#s3_validate_image_output = s3_output + "/VOCValid/JPEGImages/"
#s3_validate_annot_output = s3_output + "/VOCValid/Annotations/"
#s3_train_imageset_output = s3_output + "/VOCValid/ImageSets/Main/"

# Generate a spiral trajectory
spiral_trajectory_points = 90           # Number of points along the spiral trajectory to use as overlay anchors
theta = np.radians(np.linspace(0, 360*4, spiral_trajectory_points))
r = theta**2
spiral_trajectory_x = r*np.cos(theta)   # X coordinate of current position on spiral trajectory
spiral_trajectory_y = r*np.sin(theta)   # Y coordinate of current position on spiral trajectory
# Uncomment to see spiral trajectory
#plt.figure(figsize=[10, 10])
#plt.plot(spiral_trajectory_x, spiral_trajectory_y)
#plt.show()

target_h = 700
target_w = 700
train_set_percent = .7

points = zip(spiral_trajectory_x, spiral_trajectory_y)

#job_id = str(datetime.datetime.now())   # Unique ID + run date/time for this job
#job_id = job_id.replace(":", "-")
#job_id = job_id.replace(".", "-")
#job_id = job_id.replace(" ", "")


# pre-download all background images from s3
def create_backgrounds():
    global background_generator
    global m_length
    background_generator = backgrounds.Generator(s3_background_bucket, target_h, target_w)
    m_length = background_generator.get_count()

# Function to iterate each class and axis, move 70% to VOCTrain and 30% to VOCValid
def split(c_name, axis):
    global job_id
    global train_set_size
    global rotation_type
    try:
        for root, directory, files in os.walk('./tmp/images/' + c_name + '/' + axis):
            for img_type in directory:

                if img_type != 'renders':
                    print('img_type',img_type)
                    image_list = []
                    for r, dir, filenames in os.walk('./tmp/images/' + c_name + '/' + axis + '/' + img_type):
                        for file in filenames:
                            image_list.append(file)

                    img_source = './tmp/images/' + c_name + '/' + axis + '/' + img_type + '/'
                    annot_source = './tmp/annotations/'
                    voctrain_img_dest = './tmp/' + job_id + '/VOCTrain/JPEGImages/'
                    vocvalid_img_dest = './tmp/' + job_id + '/VOCValid/JPEGImages/'
                    voctrain_annot_dest = './tmp/' + job_id + '/VOCTrain/Annotations/'
                    vocvalid_annot_dest = './tmp/' + job_id + '/VOCValid/Annotations/'
                    voctrain_set_dest = './tmp/' + job_id + '/VOCTrain/ImageSets/Main/train_'+rotation_type+'_'+start_render_index+'_.txt'
                    vocvalid_set_dest = './tmp/' + job_id + '/VOCValid/ImageSets/Main/valid_'+rotation_type+'_'+start_render_index+'_.txt'

                    random.shuffle(image_list)
                    # Get 70% for training
                    train_set_size = round(len(image_list) * train_set_percent)

                    # Copy the train_set % VOC structure (training)
                    sev = range(0, train_set_size)
                    for i in sev:
                        file = image_list[i]
                        # Copy Image file to VOCTrain
                        shutil.copyfile(img_source + file, voctrain_img_dest + file)

                        # Copy Annot file to VOCTrain
                        file_name, ext = os.path.splitext(file)
                        shutil.copyfile(annot_source + file_name + '.xml', voctrain_annot_dest + file_name + '.xml')

                        # Write filename to train.txt
                        with open(voctrain_set_dest, 'a+') as f:
                            f.write(file_name + "\n")

                    # Copy the remaining % to VOC structure (validation)
                    thi = range(train_set_size + 1, len(image_list))
                    for i in thi:
                        file = image_list[i]
                        # Copy Image for to VOCValid
                        shutil.copyfile(img_source + file, vocvalid_img_dest + file)

                        # Copy Annot file to VOCValid
                        file_name, ext = os.path.splitext(file)
                        shutil.copyfile(annot_source + file_name + '.xml', vocvalid_annot_dest + file_name + '.xml')

                        # Write filename to valid.txt
                        with open(vocvalid_set_dest, 'a+') as f:
                            f.write(file_name + "\n")

    except Exception as err:
        print(err)


# Create the VOC annotation file for a specific image
def write_voc(annot_filename, height, width, depth, bbox, c_name):
    global annot_directory

    try:
        # Build the Structure of the VOC File
        annot = ET.Element('annotation')
        fname = ET.SubElement(annot, 'filename')
        size = ET.SubElement(annot, 'size')
        img_width = ET.SubElement(size, 'width')
        img_height = ET.SubElement(size, 'height')
        img_depth = ET.SubElement(size, 'depth')
        obj_node = ET.SubElement(annot, 'object')
        class_name_node = ET.SubElement(obj_node, 'name')
        diff = ET.SubElement(obj_node, 'difficult')
        bndbox = ET.SubElement(obj_node, 'bndbox')
        xmin_node = ET.SubElement(bndbox, 'xmin')
        ymin_node = ET.SubElement(bndbox, 'ymin')
        xmax_node = ET.SubElement(bndbox, 'xmax')
        ymax_node = ET.SubElement(bndbox, 'ymax')

        fname.text = annot_filename

        img_width.text = str(height)
        img_height.text = str(width)
        img_depth.text = str(depth)

        class_name_node.text = c_name
        diff.text = "0"  # TODO - find out what this even means?
        xmin_node.text = str(bbox[0])
        ymin_node.text = str(bbox[1])
        xmax_node.text = str(bbox[2])
        ymax_node.text = str(bbox[3])

        xml_str = ET.tostring(annot).decode()

        with open("./tmp/annotations/" + annot_filename, "w") as annot_file:
            annot_file.write(xml_str)

    except Exception as msg:
        logging.error("def write_voc::" + str(msg))


# Function to merge the rendered image with a random background into a single .jpg
def overlay_transparent(background, render, anchor_x, anchor_y):
    #x, y = top left corner of the render
    try:
        # Get the height and width of the background image
        background_width = background.shape[1]
        background_height = background.shape[0]

        if anchor_x >= background_width:
            print("x anchor is greater than background width")
            raise ValueError("x anchor is greater than background width")

        if anchor_y >= background_height:
            print("y anchor is greater than background height")
            raise ValueError("y anchor is greater than background height")

        h, w = render.shape[0], render.shape[1]

        if anchor_x + w > background_width:
            w = background_width - anchor_x
            render = render[:, :w]

        if anchor_y + h > background_height:
            h = background_height - anchor_y
            render = render[:h]

        if render.shape[2] < 4:
            render = np.concatenate(
                [
                    render,
                    np.ones((render.shape[0], render.shape[1], 1), dtype=render.dtype) * 255
                ],
                axis=2,
            )

        overlay_image = render[..., :3]
        mask = render[..., 3:] / 255.0
        background[anchor_y:anchor_y + h, anchor_x:anchor_x + w] = (1.0 - mask) * \
                                                     background[anchor_y:anchor_y + h, anchor_x:anchor_x + w] + \
                                                                   mask * overlay_image
        return background
    except Exception as msg:
        logging.error("def overlay_transparent::" + str(msg))


# (re)size the rendered image to fit the background
def right_size(img):
    global max_base

    if (img.shape[0] > max_base) or (img.shape[1] > max_base):
        # Resize by the largest dimension
        if img.shape[1] > max_base and img.shape[1] >= img.shape[0]:
            scale_factor = int(max_base / img.shape[1] * 100)
        else:  # re-size by height not width
            scale_factor = int(max_base / img.shape[0] * 100)
        width = int(img.shape[1] * scale_factor / 100)
        height = int(img.shape[0] * scale_factor / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return img


def resize_image(img, scale):

    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return img


# function to create images with the object off-centered in a spiral pattern on the background
def create_spiral_shift_images(c_name):
    global background_generator
    try:
        axis = ['x', 'y', 'z']
        for ax in axis:  # Iterate axis folder
            logging.info("iterating {} axis".format(ax))
            file_index = 1
            for root, dir, files in os.walk("./tmp/images/" + c_name + "/" + ax + "/renders"):  # Iterate each image
                for filename in files:  # Should be 359

                    # Read the previously rendered image from disk
                    rendered_img = cv2.imread("./tmp/images/" + c_name + "/" + ax + "/renders/" + filename,
                                              cv2.IMREAD_UNCHANGED)

                    rendered_img = right_size(rendered_img)

                    # we will now change the size of the base render and create images along the spiral trajectory for each size
                    # x-small   small   medium
                    # 30%       50%     75%                   # Percent of background H & W

                    x_small = .30
                    small = .50
                    medium = .75

                    sizes = [x_small, small, medium]

                    # This will result in a new image for each render at each different size (3) and at each different
                    # point on the spiral trajectory (3). Thus 359 * 3 * 3 unique images per axis will be created
                    size_index = 1
                    for size in sizes:

                        resized_img = resize_image(rendered_img, size)

                        overlay_w = resized_img.shape[1]
                        overlay_h = resized_img.shape[0]

                        spiral_index = 1
                        # size the image accordingly and then sent it on spiral trajectory
                        for p in range(spiral_trajectory_points + 1):
                            # Plot an image every 20 trajectory points - should be 4 images of each size, each orientation
                            if p > 0 and p % 22 == 0:
                                # get a random background image from background catalog
                                key = random.randrange(0, m_length)
                                bckgrnd = cv2.cvtColor(background_generator.get_background(key), cv2.COLOR_RGB2BGR)

                                # Get the center point of the background image
                                center_x = target_w / 2
                                center_y = target_h / 2

                                # logging.debug("Background center point = {},{}".format(center_x, center_y))
                                #print("Background center point = {},{}".format(center_x, center_y))

                                # Get the point along the spiral trajectory
                                spiral_x = int(spiral_trajectory_x[p])  # Round by casting to int8
                                spiral_y = int(spiral_trajectory_y[p])  # Round by casting to int8

                                # Spiral point is based on 0,0 being in the center of the background image.
                                # Base image 0,0 is the top left corner
                                # we need to modify center_x & center_y point based on spiral_x and spiral_y.

                                # logging.debug("Spiral trajectory point = {},{}".format(spiral_x, spiral_y))

                                x_min = 0
                                y_min = 0

                                # X
                                # Top left corner of the rendered image = spiral point
                                if spiral_x == 0:  # Spiral point and background center point are both equal at zero
                                    x_min = int(center_x)
                                elif spiral_x > 0:  # Spiral point is to the right of background center
                                    x_min = int(center_x + spiral_x)
                                else:  # Spiral point is to the left of background center
                                    x_min = int(center_x - abs(spiral_x))

                                # Make sure 100% of the overlay is showing on X axis (positive width)
                                if x_min >= target_w - overlay_w:
                                    # bump it back so 100% of the object is visible on background
                                    x_min = int(target_w - overlay_w)
                                # Make sure 100% of the overlay is showing on the X axis (negative width)
                                if x_min < 0:
                                    x_min = 1

                                # Y
                                if spiral_y == 0:
                                    y_min = int(center_y)
                                elif spiral_y > 0:  #
                                    y_min = int(center_y + spiral_y)
                                else:
                                    y_min = int(center_y - abs(spiral_y))

                                # Make sure it did not move beyond the height of background
                                if y_min >= target_h - overlay_h:
                                    # bump it back so at least 25% of the object is visible on background
                                    y_min = int(target_h - overlay_h)

                                if y_min < 0:
                                    y_min = 1

                                # logging.debug("Top left corner of render = {}, {}".format(x_min, y_min))
                                #print("Top left corner of render = {}, {}".format(x_min, y_min))

                                # Merge the background and rendered image
                                final_img = overlay_transparent(bckgrnd, resized_img, x_min, y_min)

                                # ######
                                # Update the bounding box info for the newly created image
                                # x_min, y_min is now the top left corner of the bounding box
                                x_max = int(x_min + resized_img.shape[1])
                                y_max = int(y_min + resized_img.shape[0])

                                if x_max > target_w: x_max = target_w
                                if y_max > target_h: y_max = target_h

                                new_box = [x_min, y_min, x_max, y_max]

                                str_index = str(file_index)
                                if len(str_index) == 1:
                                    str_index = "00" + str_index
                                elif len(str_index) == 2:
                                    str_index = "0" + str_index

                                # write the final .JPG to disk
                                diskname = str_index + "-" + str(size_index) + "-" + str(spiral_index) + "_" + c_name + '_' + ax + "_ss"
                                cv2.imwrite("./tmp/images/" + c_name + "/" + ax + "/ss/" + diskname + ".jpg", final_img)

                                # write the VOC File
                                voc_file = diskname + ".xml"
                                write_voc(voc_file, target_h, target_w, 3, new_box, c_name)

                                spiral_index += 1

                        size_index += 1

                    file_index += 1

    except Exception as err:
        logging.error("def create_spiral_shift_images:: {}".format(err))


# Flip the base image horizontally and vertically
def create_mirror_images(c_name, direction):
    axis = ['x', 'y', 'z']
    for ax in axis:
        file_index = 1
        for root, dir, files in os.walk("./tmp/images/" + c_name + "/" + ax + "/renders"):
            for filename in files:
                # Read the rendered image from disk
                new_img = cv2.imread("./tmp/images/" + c_name + "/" + ax + "/renders/" + filename, cv2.IMREAD_UNCHANGED)

                # Images may be rendered larger than the background image of 700x700 to preserve detail
                # resize the new_img to 600 max height or width as the baseline - if needed
                new_img = right_size(new_img)

                # get a random background image from background catalog
                key = random.randrange(0, m_length)
                bckgrnd = cv2.cvtColor(background_generator.get_background(key), cv2.COLOR_RGB2BGR)

                # Find center point of background image
                x = target_w / 2
                y = target_h / 2

                # find center point of rendered image
                xx = new_img.shape[1] / 2
                yy = new_img.shape[0] / 2

                # Subtract center point (X,Y) of rendered image from the center point (X, Y) of the background image
                # to position the rendered image in the center of the background
                x_min = int(round(x - xx))
                y_min = int(round(y - yy))

                if direction == 'vertical':
                    # Flip the image vertically
                    new_img = cv2.flip(new_img, 0)
                else:
                    # Flip the image horizontally
                    new_img = cv2.flip(new_img, 1)

                # Merge the background and rendered image
                final_img = overlay_transparent(bckgrnd, new_img, x_min, y_min)

                # ######
                # Update the bounding box info for the newly created image
                # ######
                # x_min, y_min is now the top left corner of the bounding box
                new_box = [x_min, y_min, int(x_min + new_img.shape[1]), int(y_min + new_img.shape[0])]

                str_index = str(file_index)
                if len(str_index) == 1:
                    str_index = "00" + str_index
                elif len(str_index) == 2:
                    str_index = "0" + str_index

                try:
                    # write the final .JPG to disk
                    if direction == 'vertical':
                        diskname = str_index + "_v_" + c_name + "_" + ax + "_base"
                    else:
                        diskname = str_index + "_h_" + c_name + "_" + ax + "_base"

                    cv2.imwrite("./tmp/images/" + c_name + "/" + ax + "/base/" + diskname + ".jpg", final_img)

                    # write the VOC File
                    voc_file = diskname + ".xml"
                    write_voc(voc_file, target_h, target_w, 3, new_box, c_name)
                except Exception as err:
                    logging.error("def create_base_image::{}".format(err))
                file_index += 1


# function to create a base set of images based on the rendered image and a background
def create_base_images(c_name):
        # Bounding_box[0] = min_x
        # Bounding_box[1] = min_y
        # Bounding_box[2] = max_x (width)
        # Bounding_box[3] = max_y (height)
        axis = ['x', 'y', 'z']
        for ax in axis:
            file_index = 1
            for root, dir, files in os.walk("./tmp/images/" + c_name + "/" + ax + "/renders"):
                for filename in files:

                    # Read the rendered image from disk
                    new_img = cv2.imread("./tmp/images/" + c_name + "/" + ax + "/renders/" + filename,
                                         cv2.IMREAD_UNCHANGED)

                    # Images may be rendered larger than the background image of 700x700 to preserve detail
                    # resize the new_img to 600 max height or width as the baseline - if needed
                    new_img = right_size(new_img)

                    # get a random background image from background catalog
                    key = random.randrange(0, m_length)
                    bckgrnd = cv2.cvtColor(background_generator.get_background(key), cv2.COLOR_RGB2BGR)

                    # ########
                    # Merge the cropped image with a random background
                    # ########
                    logging.debug("Merging rendered image with background")

                    # Find center point of background image
                    x = target_w / 2
                    y = target_h / 2

                    # find center point of rendered image
                    xx = new_img.shape[1] / 2
                    yy = new_img.shape[0] / 2

                    # Subtract center point (X,Y) of rendered image from the center point (X, Y) of the background image
                    # to position the rendered image in the center of the background
                    x_min = int(round(x - xx))
                    y_min = int(round(y - yy))

                    # Merge the background and rendered image
                    final_img = overlay_transparent(bckgrnd, new_img, x_min, y_min)

                    # ######
                    # Update the bounding box info for the newly created image
                    # ######
                    # x_min, y_min is now the top left corner of the bounding box
                    new_box = [x_min, y_min, int(x_min + new_img.shape[1]), int(y_min + new_img.shape[0])]

                    str_index = str(file_index)
                    if len(str_index) == 1:
                        str_index = "00" + str_index
                    elif len(str_index) == 2:
                        str_index = "0" + str_index

                    try:
                        # write the final .JPG to disk
                        diskname = str_index + "_" + c_name + "_" + ax + "_base"
                        cv2.imwrite("./tmp/images/" + c_name + "/" + ax + "/base/" + diskname + ".jpg", final_img)

                        # write the VOC File
                        voc_file = diskname + ".xml"
                        write_voc(voc_file, target_h, target_w, 3, new_box, c_name)

                    except Exception as err:
                        logging.error("def create_base_image::{}".format(err))
                    file_index += 1


def main():

    create_backgrounds()
    classes = []

    for root, dirs, files in os.walk('./tmp'):
        # only process the .obj files for now
        for filename in files:
            class_name, ext = os.path.splitext(filename)
            if ext.lower() == '.obj':
                classes.append(class_name)

    for c in classes:
        print("Generating base images for {} class".format(c))
        create_base_images(c)
        #print("Generating scaled/shifted images for {} class".format(c))
        create_mirror_images(c, 'vertical')
        create_mirror_images(c, 'horizontal')
        create_spiral_shift_images(c)

if __name__ == '__main__':
    global job_id
    global rotation_type

    print('augment_engine:',sys.argv)
    argv = sys.argv
    argv = argv[argv.index("--")+1:] # Get all the args after "--"
    print('augment_engine::',argv)

    job_id = argv[4]
    print('augment_engine:job_id',job_id)    
    
    rotation_type = argv[2]
    print("rotation_type = {} (X/Y/Z/A)|For generating full dataset, specify A".format(rotation_type))
    start_render_index = argv[5]
    print("start_render_index = {} (0-360)|For generating full dataset, specify 0".format(start_render_index))

    logging.info("******************************")
    logging.info("New ODIN augment_engine session started job-id: {}".format(job_id))
    logging.info("******************************")

    main()