################
#blender script#
# Script uses local disk for image generation and then moves images to s3
# burnsca@amazon.com
################

import math
from mathutils import *
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
job_id = str(datetime.datetime.now())
job_id = job_id.replace(":", "-")
job_id = job_id.replace(".", "-")
job_id = job_id.replace(" ", "")

s3_bucket = ''                          # Bucket where .obj files can be found for processing
s3_background_bucket = 'odin-bck'       # Bucket where background images are stored

# Output
s3_output = s3_bucket + '/output/' + job_id + '/VOC'

# Training
s3_train_image_output = s3_output + "/VOCTrain/JPEGImages/"
s3_train_annot_output = s3_output + "/VOCTrain/Annotations/"
s3_train_imageset_output = s3_output + "/VOCTrain/ImageSets/Main/"

# Validation
s3_validate_image_output = s3_output + "/VOCValid/JPEGImages/"
s3_validate_annot_output = s3_output + "/VOCValid/Annotations/"
s3_train_imageset_output = s3_output + "/VOCValid/ImageSets/Main/"

# TODO - add the below variables to argparse
theta = 1    # amount (in degrees) to rotate the object - on each axis - for each render
upper_bound = 360
scale_factor = 3
target_h = 700
target_w = 700

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
        os.mkdir('./tmp/renders/')
        os.mkdir('./tmp/backgrounds/')
        os.mkdir('./tmp/final/')
        os.mkdir('./tmp/final/JPEGImages')
        os.mkdir('./tmp/final/Annotations')
    except Exception as err:
        logging.error(err)



def split_dataset():
    # All images are in ./tmp/final/JPEGImages
    # Count images and create dict
    # Shuffle images
    # Take 70% and create ImageSet list for training
    # Take remainder and create ImageSet list for validation
    # move Training images to S3
    # move Validation images to S3
    image_list = []
    for root, folder, files in os.walk("./tmp/final/JPEGImages/"):
        for filename in files:
            image_list.append(filename)


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
    logging.info("Obtaining random background image from S3")
    m_file = 'manifest.txt'
    m_length = 5640  # Num of entries in Manifest file - update this if file is altered
    m_index = {}
    s3 = boto3.client("s3")
    try:
        # Gen random number
        key = random.randrange(0, m_length+1)

        # Load manifest file
        with open(m_file) as f:
            for i, l in enumerate(f):
                m_index[i] = l.strip()

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

        with open("./tmp/final/Annotations/" + annot_filename, "w") as annot_file:
            annot_file.write(xml_str)

    except Exception as msg:
        logging.error("def write_voc::" + str(msg))


def overlay_transparent(background, overlay, x, y):

    try:
        background_width = background.shape[1]
        background_height = background.shape[0]

        if x >= background_width or y >= background_height:
            return background

        h, w = overlay.shape[0], overlay.shape[1]

        if x + w > background_width:
            w = background_width - x
            overlay = overlay[:, :w]

        if y + h > background_height:
            h = background_height - y
            overlay = overlay[:h]

        if overlay.shape[2] < 4:
            overlay = np.concatenate(
                [
                    overlay,
                    np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
                ],
                axis=2,
            )

        overlay_image = overlay[..., :3]
        mask = overlay[..., 3:] / 255.0

        background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

        return background
    except Exception as msg:
        logging.error("def overlay_transparent::" + str(msg))


def crop_image(img, target_h, target_w, bbox_x_min, bbox_y_min, obj_height, obj_width):

    try:
        orig_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)

        # Height = Y axis | Width = X axis
        cur_h, cur_w, _ = orig_img.shape
        y_ax_crop = cur_h - target_h
        x_ax_crop = cur_w - target_w

        new_x_bbox_min = bbox_x_min - x_ax_crop/2
        new_y_bbox_min = bbox_y_min - y_ax_crop/2

        new_x_min = int(x_ax_crop/2)
        new_x_max = int(cur_w - (x_ax_crop/2))
        new_y_min = int(y_ax_crop/2)
        new_y_max = int(cur_h - (y_ax_crop/2))

        new_bbox = [new_x_bbox_min, new_y_bbox_min, new_x_bbox_min+obj_width, new_y_bbox_min+obj_height]

        cropped = orig_img[new_y_min:new_y_max, new_x_min:new_x_max]

        return cropped, new_bbox
    except Exception as msg:
        logging.error("def crop_image::" + str(msg))


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

        render_name = "./tmp/renders/" + c_name + "_" + str(index) + "_render_" + axis + "_{}.png".format(axis_index)

        # Render the 3D object as an image
        bpy.context.scene.render.filepath = render_name

        logging.debug("rendering pose # {} on {} axis".format(index, axis))
        # TODO EXPLORE WRITING TO A NUMPY ARRAY RATHER THAN DISK FOR BETTER PERFORMANCE
        bpy.ops.render.render(write_still=True, use_viewport=True)

        # Get the 2D bounding box for the image
        logging.debug("Generating 2D bounding box for labelling")
        bounding_box = camera_view_bounds_2d(C.scene, cam, obj)

        if os.path.exists(render_name):
            logging.debug("Cropping image prior to merge")
            # Crop the newly rendered image
            new_img, new_box = crop_image(render_name, target_h, target_w, bounding_box[0], bounding_box[1],
                                          bounding_box[3], bounding_box[2])

            # TODO: ADD RANDOMIZED SCALING
            # TODO: ADD RANDOMIZED AUGMENTATION
            # TODO: ADD RANDOMIZED PLACEMENT DURING MERGE

            # get a random background image from Lambda function
            bckgrnd = get_background()
            bckgrnd = cv2.resize(bckgrnd, (target_h, target_w))

            # Merge the cropped image with a random background
            logging.debug("Merging rendered image with background")
            final_img = overlay_transparent(bckgrnd, new_img, 0, 0)

            # write the final JPG
            cv2.imwrite("./tmp/final/JPEGImages/" + c_name + "_" + str(index) + "_" + axis + "_{}.jpg".format(axis_index), final_img)

            # delete the partially rendered image
            os.remove(render_name)

            # write the VOC File
            voc_file = c_name +  "_" + str(index) + "_" + axis + "_" + str(axis_index) + ".xml"
            write_voc(voc_file, target_h, target_w, 3, new_box, c_name)
        else:
            logging.error("Rendered image is unavailable for merging!")
    except Exception as err:
        logging.error("def render:: {}".format(err))


# Orchestrator loads the .obj file into blender workspace
# If the object is independent meshes, it merges the meshes into a single mesh
# The single mesh is than scaled appropriately
# Finally the mesh is rotated by theta on each axis and an image is rendered to disk
def orchestrate(three_d_obj, c_name):
    global image_directory
    global annot_directory
    global theta
    try:
        image_index = 0

        # clear default scene
        bpy.ops.object.delete() # should delete cube

        # add object
        bpy.ops.import_scene.obj(filepath='./tmp/' + three_d_obj)

        # We don't need to join objects in Blender 2.8x
        # TODO - Determine which type of models need to be joined and which do not so we don't break this

        if len(C.scene.objects) > 3:
            # Join the objects together and set the master object to be the active object
            for ob in C.scene.objects:
                if ob.type == "MESH":
                    ob.select_set(True)
                    # TODO - this makes the assumption there is only one mesh  - could end badly
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
            angle = (start_angle * (math.pi/180)) + (z*-1) * (theta * (math.pi/180))
            render(obj, angle, "z", image_index, z, c_name)
            image_index += 1

        # Rotate on the X axis
        for x in range(1, upper_bound):
            angle = (start_angle * (math.pi/180)) + (x*-1) * (theta * (math.pi/180))
            render(obj, angle, "x", image_index, x, c_name)
            image_index += 1

        # Rotate on the Y axis
        for y in range(1, upper_bound):
            angle = (start_angle * (math.pi /180)) + (y*-1) * (theta * (math.pi/180))
            render(obj, angle, "y", image_index, y, c_name)
            image_index += 1

        # Split the dataset into train/validation
        #split_dataset()
    except Exception as err:
        logging.error("def orchestrate:: {}".format(err))


# Download relevant .obj files from s3 and call the orchestrator
def main():

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