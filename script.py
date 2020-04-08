################
#blender script#
# Script uses local disk for image generation and then moves images to s3
# burnsca@amazon.com
################

import math
from mathutils import *
from math import cos, sin, pi
from numpy import linspace
import bpy
import random
import cv2
import numpy as np
import sys
import os
import xml.etree.ElementTree as ET
import boto3
import shutil
import logging
import datetime
import time

logging.basicConfig(filename='runtime.log', level=logging.INFO)
C = bpy.context                         # Abbreviate the bpy.context namespace as it is used frequently
job_id = str(datetime.datetime.now())   # Unique ID + run date/time for this job
job_id = job_id.replace(":", "-")
job_id = job_id.replace(".", "-")
job_id = job_id.replace(" ", "")

s3_bucket = ''                          # Bucket where .obj files can be found for processing
s3_background_bucket = 'odin-bck'       # Bucket where background images are stored


spiral_trajectory_points = 90           # Number of points along the spiral trajectory to use as overlay anchors
theta = np.radians(np.linspace(0, 360*4, spiral_trajectory_points))
r = theta**2
spiral_trajectory_x = r*np.cos(theta)   # X coordinate of current position on spiral trajectory
spiral_trajectory_y = r*np.sin(theta)   # Y coordinate of current position on spiral trajectory
#plt.figure(figsize=[10, 10])
#plt.plot(spiral_trajectory_x, spiral_trajectory_y)
#plt.show()

# S3 Folder structure
s3_output = s3_bucket + '/output/' + job_id + '/VOC'
s3_train_image_output = s3_output + "/VOCTrain/JPEGImages/"
s3_train_annot_output = s3_output + "/VOCTrain/Annotations/"
s3_train_imageset_output = s3_output + "/VOCTrain/ImageSets/Main/"
s3_validate_image_output = s3_output + "/VOCValid/JPEGImages/"
s3_validate_annot_output = s3_output + "/VOCValid/Annotations/"
s3_train_imageset_output = s3_output + "/VOCValid/ImageSets/Main/"

# TODO - add the below variables to argparse
rotation_theta = 1      # amount (in degrees) to rotate the object - on each axis - for each render
upper_bound = 360       # 360 degrees of total rotation
scale_factor = 3
target_h = 700
target_w = 700
train_set_percent = .7


# Function to clear old workspace if exists and create fresh folder structure
def create_workspace():
    try:
        # Create a tmp directory to hold all objects from the s3 bucket in args
        if os.path.exists('./tmp'):
            logging.info("tmp directory exists from previous run.  deleting now")
            # delete existing tmp directory and recreate
            shutil.rmtree('./tmp')
        logging.info("Creating new /tmp directory")
        os.mkdir('./tmp')
        os.mkdir('./tmp/backgrounds/')
        os.mkdir('./tmp/annotations')
        os.mkdir('./tmp/images/')
    except Exception as err:
        logging.error(err)


def create_workspace_classes(classes):
    global job_id
    try:
        for c in classes:
            os.mkdir('./tmp/images/' + c)
            os.mkdir('./tmp/images/' + c + '/x')
            os.mkdir('./tmp/images/' + c + '/x/renders')
            os.mkdir('./tmp/images/' + c + '/x/base')               # Base image as it was rendered and merged with bck
            os.mkdir('./tmp/images/' + c + '/x/ss')                 # Scaled and shifted base image merged with bck
            os.mkdir('./tmp/images/' + c + '/x/augmented')          # Base images augmented and merged with bck

            os.mkdir('./tmp/images/' + c + '/y')
            os.mkdir('./tmp/images/' + c + '/y/renders')
            os.mkdir('./tmp/images/' + c + '/y/base')
            os.mkdir('./tmp/images/' + c + '/y/ss')
            os.mkdir('./tmp/images/' + c + '/y/augmented')

            os.mkdir('./tmp/images/' + c + '/z')
            os.mkdir('./tmp/images/' + c + '/z/renders')
            os.mkdir('./tmp/images/' + c + '/z/base')
            os.mkdir('./tmp/images/' + c + '/z/ss')
            os.mkdir('./tmp/images/' + c + '/z/augmented')

        # Directory Structure for VOC
        os.mkdir('./tmp/' + job_id)
        os.mkdir('./tmp/' + job_id + '/VOCTrain')
        os.mkdir('./tmp/' + job_id + '/VOCTrain/JPEGImages')
        os.mkdir('./tmp/' + job_id + '/VOCTrain/Annotations')
        os.mkdir('./tmp/' + job_id + '/VOCTrain/ImageSets')
        os.mkdir('./tmp/' + job_id + '/VOCTrain/ImageSets/Main')
        os.mkdir('./tmp/' + job_id + '/VOCValid')
        os.mkdir('./tmp/' + job_id + '/VOCValid/JPEGImages')
        os.mkdir('./tmp/' + job_id + '/VOCValid/Annotations')
        os.mkdir('./tmp/' + job_id + '/VOCValid/ImageSets')
        os.mkdir('./tmp/' + job_id + '/VOCValid/ImageSets/Main')
    except Exception as err:
        logging.error("def create_workspace_classes:: {}".format(err))


# Function to iterate each class and axis, move 70% to VOCTrain and 30% to VOCValid
def split(c_name, axis):
    global train_set_size
    try:
        for root, directory, files in os.walk('./tmp/images/' + c_name + '/' + axis):
            for img_type in directory:

                if img_type != 'renders':
                    print(img_type)
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
                    voctrain_set_dest = './tmp/' + job_id + '/VOCTrain/ImageSets/Main/train.txt'
                    vocvalid_set_dest = './tmp/' + job_id + '/VOCValid/ImageSets/Main/valid.txt'

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


# Get the .obj files from a specified s3 bucket
def get_s3_contents():
    global s3_bucket
    s3 = boto3.resource("s3")
    try:
        target_bucket = s3.Bucket(s3_bucket)
        logging.info("Checking for objects in: {}".format(target_bucket))

        for s3_obj in target_bucket.objects.all():
            path, filename = os.path.split(s3_obj.key)
            if not "/" in s3_obj.key:  # Don't download subfolder contents
                logging.info("Downloading: {}".format(s3_obj.key))
                target_bucket.download_file(s3_obj.key, './tmp/' + filename)
    except Exception as err:
        logging.error("def get_s3_contents:: {}".format(str(err)))


# Get a random background from a Lambda function
def get_background():

    m_file = 'manifest.txt'
    m_length = 5640  # Num of entries in Manifest file - update this if file is altered
    m_index = {}
    s3 = boto3.client("s3")
    try:
        # Gen random number
        key = random.randrange(0, m_length+1)

        # Load manifest file to select random background based on key
        with open(m_file) as f:
            for i, l in enumerate(f):
                m_index[i] = l.strip()

        # Check if background has already been downloaded:
        if not os.path.exists('./tmp/backgrounds/' + m_index[key]):
            # Download corresponding object from S3
            with open("./tmp/backgrounds/" + m_index[key], 'wb') as f:
                s3.download_fileobj(s3_background_bucket, m_index[key], f)

        img = cv2.imread("./tmp/backgrounds/" + m_index[key])

        return img

    except Exception as err:
        logging.error("def get_background::" + str(err))


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
        #logging.error("def overlay_transparent::" + str(msg))
        print(msg)


def crop_image(img, min_x, max_x, min_y, max_y):
    cropped = img[min_y:max_y, min_x:max_x]
    return cropped


# Helper function for camera_view_bounds_2d
def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


# Function to calculate the 2D bounding box of the object that was just rendered
def camera_view_bounds_2d(scene, cam_ob, me_ob):
    try:
        mat = cam_ob.matrix_world.normalized().inverted()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mesh_eval = me_ob.evaluated_get(depsgraph)
        me = mesh_eval.to_mesh()
        me.transform(me_ob.matrix_world)
        me.transform(mat)
        camera = cam_ob.data
        frame = [-v for v in camera.view_frame(scene=scene)[:3]]
        camera_persp = camera.type != 'ORTHO'

        lx = []
        ly = []

        for v in me.vertices:
            co_local = v.co
            z = -co_local.z

            if camera_persp:
                if z == 0.0:
                    lx.append(0.5)
                    ly.append(0.5)
                # Does it make any sense to drop these?
                # if z <= 0.0:
                #    continue
                else:
                    frame = [(v / (v.z / z)) for v in frame]

            min_x, max_x = frame[1].x, frame[2].x
            min_y, max_y = frame[0].y, frame[1].y

            x = (co_local.x - min_x) / (max_x - min_x)
            y = (co_local.y - min_y) / (max_y - min_y)

            lx.append(x)
            ly.append(y)

        min_x = clamp(min(lx), 0.0, 1.0)
        max_x = clamp(max(lx), 0.0, 1.0)
        min_y = clamp(min(ly), 0.0, 1.0)
        max_y = clamp(max(ly), 0.0, 1.0)

        mesh_eval.to_mesh_clear()

        r = scene.render
        fac = r.resolution_percentage * 0.01
        dim_x = r.resolution_x * fac
        dim_y = r.resolution_y * fac

        # Sanity check
        if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
            return 0, 0, 0, 0

        return (
            round(min_x * dim_x),            # X
            round(dim_y - max_y * dim_y),    # Y
            round((max_x - min_x) * dim_x),  # Width
            round((max_y - min_y) * dim_y)   # Height
        )
    except Exception as msg:
        logging.error("def camera_view_bounds_2d::" + str(msg))


# Render function will download a random background from S3
# merge the newly rendered image with the random background and write to disk
# create and write an accompanying VOC file for the newly rendered image
def render(obj, angle, axis, index, axis_index, c_name):
    try:
        cam = bpy.data.objects['Camera']

        if axis == "z":
            obj.rotation_euler = (0, 0, angle)
        elif axis == "y":
            obj.rotation_euler = (0, angle, 0)
        else:
            obj.rotation_euler = (angle, 0, 0)

        # Get the 2D bounding box for the image
        logging.debug("Generating 2D bounding box for labelling")
        bounding_box = camera_view_bounds_2d(C.scene, cam, obj)

        render_name = "./tmp/images/" + c_name + "/" + axis + "/renders/" + axis + "_{}-{}-{}-{}-{}.png".format(
            axis_index, bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3])

        # Render the 3D object as an image
        bpy.context.scene.render.filepath = render_name

        logging.debug("rendering pose # {} on {} axis".format(index, axis))
        # TODO EXPLORE WRITING TO A NUMPY ARRAY RATHER THAN DISK FOR BETTER PERFORMANCE
        bpy.ops.render.render(write_still=True, use_viewport=True)

        # Rendering sends the entire scene to an image file. Crop it down to our bounding boxes
        # After the crop, the rendered image size IS the bounding box.
        if os.path.exists(render_name):
            img = cv2.imread(render_name, cv2.IMREAD_UNCHANGED)
            min_x = int(bounding_box[0])
            min_y = int(bounding_box[1])
            max_x = min_x + int(bounding_box[2])
            max_y = min_y + int(bounding_box[3])
            cropped = img[min_y:max_y, min_x:max_x]

            # Remove the original render from disk
            os.remove(render_name)
            # Write the newly cropped render to disk
            cv2.imwrite(render_name, cropped)
    except Exception as err:
        logging.error("def render:: {}".format(err))


# function to create a base set of images based on the rendered image and a background
def create_base_images(c_name):
    try:
        # Bounding_box[0] = min_x
        # Bounding_box[1] = min_y
        # Bounding_box[2] = max_x (width)
        # Bounding_box[3] = max_y (height)
        axis = ['x', 'y', 'z']
        for ax in axis:
            for root, dir, files in os.walk("./tmp/images/" + c_name + "/" + ax + "/renders"):
                for filename in files:
                    # Pull data out of the filename
                    data, ext = os.path.splitext(filename)

                    elements = data.split("-")
                    a = elements[0]
                    bounding_box = [elements[1], elements[2], elements[3], elements[4]]

                    # Read the rendered image from disk
                    new_img = cv2.imread("./tmp/images/" + c_name + "/" + ax + "/renders/" + filename,
                                         cv2.IMREAD_UNCHANGED)

                    # get a random background image from S3 or local cache
                    bckgrnd = get_background()
                    bckgrnd = cv2.resize(bckgrnd, (target_h, target_w))

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

                    # write the final .JPG to disk
                    diskname = c_name + "_" + a + "_base"
                    cv2.imwrite("./tmp/images/" + c_name + "/" + ax + "/base/" + diskname + ".jpg", final_img)

                    # write the VOC File
                    voc_file = diskname + ".xml"
                    write_voc(voc_file, target_h, target_w, 3, new_box, c_name)

    except Exception as err:
        logging.error("def create_base_image::{}".format(err))


# function to create images with the object off-centered in a spiral pattern on the background
def create_spiral_shift_images(c_name):
    global spiral_trajectory_points
    spiral_iteration = 1  # current number of steps taken along the spiral trajectory
    # Bounding_box[0] = min_x
    # Bounding_box[1] = min_y
    # Bounding_box[2] = max_x (width)
    # Bounding_box[3] = max_y (height)
    try:
        axis = ['x', 'y', 'z']
        file_index = -1
        for ax in axis:
            for root, dir, files in os.walk("./tmp/images/" + c_name + "/" + ax + "/renders"):
                for filename in files:  # Should be 359
                    # Pull data out of the filename
                    data, ext = os.path.splitext(filename)

                    elements = data.split("-")
                    a = elements[0]
                    bounding_box = [elements[1], elements[2], elements[3], elements[4]]

                    # Read the rendered image from disk
                    new_img = cv2.imread("./tmp/images/" + c_name + "/" + ax + "/renders/" + filename,
                                         cv2.IMREAD_UNCHANGED)

                    overlay_w = new_img.shape[1]
                    overlay_h = new_img.shape[0]

                    # get a random background image from S3
                    bckgrnd = get_background()
                    bckgrnd = cv2.resize(bckgrnd, (target_h, target_w))

                    # Get the center point of the background image
                    center_x = target_w / 2
                    center_y = target_h / 2

                    logging.debug("Background center point = {},{}".format(center_x, center_y))

                    # Get the spiral trajectory coordinates
                    spiral_x = int(spiral_trajectory_x[spiral_iteration])   # Round by casting to int8
                    spiral_y = int(spiral_trajectory_y[spiral_iteration])   # Round by casting to int8

                    # Spiral point is based on 0,0 being in the center of the background image.
                    # we need to modify center_x & center_y point based on spiral_x and spiral_y.

                    logging.debug("Spiral trajectory point = {},{}".format(spiral_x, spiral_y))

                    x_min = 0
                    y_min = 0

                    # X
                    # Top left corner of the rendered image = spiral point
                    if spiral_x == 0:   # Spiral point and background center point are equal at zero
                        x_min = int(center_x)
                    elif spiral_x > 0:  # Spiral point is to the right of background center
                        x_min = int(center_x + spiral_x)
                    else:               # Spiral point is to the left of background center
                        x_min = int(center_x - abs(spiral_x))

                    two_five = int(overlay_w * .25)
                    seventy_five = int(overlay_w * .75)

                    # Make sure at least 25% of the overlay is showing in positive width
                    if x_min >= target_w - two_five:
                        # bump it back so at least 25% of the object is visible on background
                        x_min = int(target_w - two_five)

                    if x_min < 0:
                        x_min = 1

                    # Y
                    if spiral_y == 0:
                        y_min = int(center_y)
                    elif spiral_y > 0: #
                        y_min = int(center_y + spiral_y)
                    else:
                        y_min = int(center_y - abs(spiral_y))

                    two_five = int(overlay_h * .25)
                    seventy_five = int(overlay_h * .75)

                    # Make sure it did not move beyond the height of background
                    if y_min >= target_h - two_five:
                        # bump it back so at least 25% of the object is visible on background
                        y_min = int(target_h - two_five)

                    if y_min < 0:
                        y_min = 1

                    logging.debug("Top left corner of render = {}, {}".format(x_min, y_min))


                    # Merge the background and rendered image
                    final_img = overlay_transparent(bckgrnd, new_img, x_min, y_min)

                    # ######
                    # Update the bounding box info for the newly created image
                    # TODO - if bounding box exceeds target_h or target_w adjust bounding box to edges of target
                    # ######
                    # x_min, y_min is now the top left corner of the bounding box
                    x_max = int(x_min + new_img.shape[1])
                    y_max = int(y_min + new_img.shape[0])

                    if x_max > target_w: x_max = target_w
                    if y_max > target_h: y_max = target_h

                    new_box = [x_min, y_min, x_max, y_max]

                    # write the final .JPG to disk
                    file_index += 1
                    diskname = c_name + '_' + str(file_index) + "_ss"
                    cv2.imwrite("./tmp/images/" + c_name + "/" + ax + "/ss/" + diskname + ".jpg", final_img)

                    # write the VOC File
                    voc_file = diskname + ".xml"
                    write_voc(voc_file, target_h, target_w, 3, new_box, c_name)

                    spiral_iteration += 1
                    if spiral_iteration == spiral_trajectory_points:
                        spiral_iteration = 0        # Reset overlay anchor to go through the spiral again.

    except Exception as err:
        logging.error("def create_spiral_shift_images:: {}".format(err))


# Orchestrator loads the .obj file into blender workspace, thus orchestrate runs for each class
# If the object is independent meshes, it merges the meshes into a single mesh
# The single mesh is than scaled appropriately
# Finally the mesh is rotated by theta on each axis and an image is rendered to disk
def orchestrate(three_d_obj, c_name):
    global rotation_theta
    try:
        image_index = 0

        # clear default scene
        bpy.ops.object.delete() # should delete cube

        # add object
        bpy.ops.import_scene.obj(filepath='./tmp/' + three_d_obj)

        # TODO - Determine which type of models need to be joined and which do not so we don't break this

        if len(C.scene.objects) > 3:
            # Join the objects together and set the master object to be the active object
            for ob in C.scene.objects:
                if ob.type == "MESH":
                    ob.select_set(True)
                    # TODO - this makes the assumption there is only one mesh  - it could end badly
                    bpy.context.view_layer.objects.active = ob
                else:
                    ob.select_set(False)
            bpy.ops.object.join()
        else:
            # Set the object to be the active object
            for ob in C.scene.objects:
                if ob.type == "MESH":
                    ob.select_set(True)
                    C.view_layer.objects.active = ob

        #  Set the image background to be transparent
        C.scene.render.film_transparent = True

        #  scale object
        s = 2/max(C.active_object.dimensions)

        C.active_object.scale = (s, s, s)

        #  rotate and render
        obj = C.active_object
        obj.rotation_mode = 'XYZ'
        start_angle = 0

        # Rotate on the Zed axis
        for z in range(1, upper_bound):
            angle = (start_angle * (math.pi/180)) + (z*-1) * (rotation_theta * (math.pi/180))
            render(obj, angle, "z", image_index, z, c_name)
            image_index += 1

        # Rotate on the X axis
        for x in range(1, upper_bound):
            angle = (start_angle * (math.pi/180)) + (x*-1) * (rotation_theta * (math.pi/180))
            render(obj, angle, "x", image_index, x, c_name)
            image_index += 1

        # Rotate on the Y axis
        for y in range(1, upper_bound):
            angle = (start_angle * (math.pi/180)) + (y*-1) * (rotation_theta * (math.pi/180))
            render(obj, angle, "y", image_index, y, c_name)
            image_index += 1

        create_base_images(c_name)
        # TODO: ADD RANDOMIZED SCALING
        # TODO: ADD RANDOMIZED AUGMENTATION
        create_spiral_shift_images(c_name)
        #create_augmented_images()

        # Split the dataset into train/validation
        split(c_name, 'x')
        split(c_name, 'y')
        split(c_name, 'z')
    except Exception as err:
        logging.error("def orchestrate:: {}".format(err))


# Download relevant .obj files from s3 and call the orchestrator
def main():
    classes = []
    try:
        start = time.time()

        create_workspace()

        # Download contents of s3 bucket (objects only, not subfolders)
        get_s3_contents()

        if not os.listdir('./tmp'):
            logging.error("Nothing downloaded from s3 - nothing to do!")
        else:
            for root, dirs, files in os.walk('./tmp'):
                # only process the .obj files for now
                for filename in files:
                    class_name, ext = os.path.splitext(filename)
                    if ext.lower() == '.obj':
                        classes.append(class_name)

            create_workspace_classes(classes)

            for root, dirs, files in os.walk('./tmp'):
                # only process the .obj files for now
                for filename in files:
                    class_name, ext = os.path.splitext(filename)
                    if ext.lower() == '.obj':
                        logging.info("processing file: {}".format(filename))
                        orchestrate(filename, class_name)

        finish = time.time()
        logging.info("dataset creation elapsed time: {}".format(finish - start))
    except Exception as err:
        logging.error("def main:: " + str(err))


if __name__ == '__main__':
    logging.info("******************************")
    logging.info("New ODIN session started job-id: {}".format(job_id))
    logging.info("******************************")

    argv = sys.argv
    argv = argv[argv.index("--")+1:] # Get all the args after "--"
    print(argv)
    theta = int(sys.argv[-1])
    logging.info("theta set to {}".format(theta))
    s3_bucket = sys.argv[-2]
    logging.info("s3_bucket = {}".format(s3_bucket))
    main()