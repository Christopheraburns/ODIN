################
#blender script#
################

import math
from mathutils import *
import bpy
import pickle
import random
import cv2
import numpy as np
import time
import os
import xml.etree.ElementTree as ET
from pathlib import Path

class_name = ""
output_folder = "data/intermediate/"
logfile = "/home/chris/Desktop/logfile.txt"
upper_bound = 360
scale_factor = 3
image_directory = ""
annot_directory = ""
target_h = 700
target_w = 700
C = bpy.context

class Backgrounds():
    def __init__(self, bckgrnd="/data/backgrounds.pck"):
        self._images = pickle.load(open(bckgrnd, 'rb'))
        self._nb_images = len(self._images)

    def get_random(self, display=False):
        bg=self._images[random.randint(0, self._nb_images - 1)]
        return bg


def write_voc(annot_filename, height, width, depth, bbox):
    global annot_directory
    global class_name

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

        class_name_node.text = class_name
        diff.text = "0"  # TODO - what does this even mean?
        xmin_node.text = str(bbox[0])
        ymin_node.text = str(bbox[1])
        xmax_node.text = str(bbox[2])
        ymax_node.text = str(bbox[3])

        xml_str = ET.tostring(annot).decode()

        annot_file = open(os.path.join(annot_directory, annot_filename), "w")
        annot_file.write(xml_str)

    except Exception as msg:
        with open(logfile, "w") as f:
            f.write("def write_voc::" + str(msg))


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
        with open(logfile, "w") as f:
            f.write("def overlay_transparent::" + str(msg))


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
        with open(logfile, "w") as f:
            f.write("def crop_image::" + str(msg))


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


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
        with open(logfile, "w") as f:
            f.write("def camera_view_bounds_2d::" + str(msg))


def render(obj, angle, axis, index, axis_index):
    '''
    obj: the 3D object to be rendered
    angle: angle to position object
    axis: which axis to apply the angle
    index: index for unique filename
    axis_index: index for axis
    Render the image to .png and write to persistance store
    '''
    cam = bpy.data.objects['Camera']

    if axis == "z":
        obj.rotation_euler = (0, 0, angle)
    elif axis == "y":
        obj.rotation_euler = (0, angle, 0)
    else:
        obj.rotation_euler = (angle, 0, 0)

    render_name = output_folder + "/" + str(index) + "_render_" + axis + "_{}.png".format(axis_index)

    # Render the 3D object as an image
    bpy.context.scene.render.filepath = render_name

    print("rendering pose # {} on {} axis".format(index, axis))
    # TODO EXPLORE WRITING TO A NUMPY ARRAY RATHER THAN DISK
    bpy.ops.render.render(write_still=True, use_viewport=True)

    # Get the 2D bounding box for the image
    bounding_box = camera_view_bounds_2d(C.scene, cam, obj)

    print("generating random background")
    # get a random background image
    bckgrnd = backgrounds.get_random()
    bckgrnd = cv2.resize(bckgrnd, (target_h, target_w))

    if os.path.exists(render_name):
        # Crop the image, layer image on the random background and get updated bounding box
        new_img, new_box = crop_image(render_name, target_h, target_w, bounding_box[0], bounding_box[1],
                                      bounding_box[3], bounding_box[2])

        # write the base image to disk
        cv2.imwrite(image_directory + "/" + str(index) + "_" + axis + "_{}.png".format(axis_index),new_img)

        # write the bounding box info to disk with the same filename as the base image
        with open(image_directory + "/" + str(index) + "_" + axis + "_{}.txt".format(axis_index), "w") as f:
            f.write(str(new_box))


    else:
        print("rendered image is unavailable for merging...")


def orchestrate(three_d_obj):
    global image_directory
    global annot_directory
    try:
        image_index = 0

        # clear default scene
        bpy.ops.object.delete() # should delete cube

        # add object
        bpy.ops.import_scene.obj(filepath=three_d_obj)

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
        theta = 1  # degrees to turn per rotation
        start_angle = 0

        # Rotate on the Z axis
        for z in range(1, upper_bound):
            angle = (start_angle * (math.pi/180)) + (z*-1) * (theta * (math.pi/180))
            render(obj, angle, "z", image_index, z)
            image_index += 1

        # Rotate on the X axis
        for x in range(1, upper_bound):
            angle = (start_angle * (math.pi/180)) + (x*-1) * (theta * (math.pi/180))
            render(obj, angle, "x", image_index, x)
            image_index += 1

        # Rotate on the Y axis
        for y in range(1, upper_bound):
            angle = (start_angle * (math.pi /180)) + (y*-1) * (theta * (math.pi/180))
            render(obj, angle, "y", image_index, y)
            image_index += 1

    except Exception as msg:
        with open(logfile, "w") as f:
            f.write(str(msg))


def main():
    global class_name
    global image_directory
    global annot_directory

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if not os.path.exists(os.path.join(output_folder, "Annotations")):
        os.mkdir(os.path.join(output_folder, "Annotations"))

    if not os.path.exists(os.path.join(output_folder, "ImageSets")):
        os.mkdir(os.path.join(output_folder, "ImageSets"))

    if not os.path.exists(os.path.join(output_folder, "JPEGImages")):
        os.mkdir(os.path.join(output_folder, "JPEGImages"))

    image_directory = os.path.join(output_folder, "JPEGImages")
    annot_directory = os.path.join(output_folder, "Annotations")

    # TODO - Convert this to a DynamoDB table query to get S3 URL of Objects
    three_d_obj = "/home/chris/Downloads/skidsteer.obj"
    class_name = "skidsteer"
    orchestrate(three_d_obj)


if __name__ == '__main__':
    #if not os.path.exists(output_folder):
    #    os.mkdir(output_folder)

    backgrounds = Backgrounds()
    time.sleep(3)
    main()


# Merge the cropped image with a random background
# final_img = overlay_transparent(bckgrnd, new_img, 0, 0)

# write the final JPG
# cv2.imwrite(image_directory + "/" + str(index) + "_" + axis + "_{}.jpg".format(axis_index),final_img)

# delete the partially rendered image
# os.remove(render_name)

# write the VOC File
# voc_file = str(index) + "_" + axis + "_" + str(axis_index) + ".xml"
# write_voc(voc_file, target_h, target_w, 3, new_box)