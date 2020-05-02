# The euler_engine leverages Blender 2.8x to load .obj (+.mtl) files from S3.  Right size the objects and then render
# a transparent .png file on each degree of rotation on each axis (X * 359 + Y * 359 + Z * 359)
# !!!! NOTE !!!! This module must be run within the Blender python environment.
# !!!! NOTE !!!! Within the python blender environment, you cannot call functions or classes in other python modules
#                unless they are install via PIP
# burnsca@

import math
from mathutils import *
import bpy
import random
import cv2
import sys
import os
import boto3
import shutil
import logging
import time
import datetime

logging.basicConfig(filename='runtime.log', level=logging.INFO)
C = bpy.context                         # Abbreviate the bpy.context namespace as it is used frequently
s3_bucket = ''                          # Bucket where .obj files can be found for processing

# TODO - add the below variables to argparse
target_h = 700
target_w = 700
max_scale = 3.25

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


# Create the file structure for class images
def create_classes_workspace(classes):
    path = './tmp/images'
    axis = ['x', 'y', 'z']
    subs = ['renders', 'base', 'ss']
    try:
        if not os.path.exists(path):
            os.mkdir(path)
        for c in classes:
            os.mkdir(path + '/' + c)
            for a in axis:
                os.mkdir(path + '/' + c + '/' + a)
                for s in subs:
                    os.mkdir(path + "/" + c + "/" + a + "/" + s)
    except Exception as err:
        logging.error("def create_classes_workspace:: {}".format(err))


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


# Function to right size the Mesh to best fit the camera viewport
def right_size(vector):
    x = vector[0]
    y = vector[1]
    z = vector[2]

    scale_factor = 1.0

    try:
        dim = max(vector)

        # Which axis is max
        if x == dim:
            if x > max_scale:
                scale_factor = max_scale / x
        elif y == dim:
            if y > max_scale:
                scale_factor = max_scale / y
        elif z == dim:
            if z > max_scale:
                scale_factor = max_scale / z

    except Exception as err:
        logging.error("def right_size:: {}".format(err))

    return scale_factor


# Render function will download a random background from S3
# merge the newly rendered image with the random background and write to disk
# create and write an accompanying VOC file for the newly rendered image
def render(obj, angle, axis, axis_index, c_name):
    try:
        cam = bpy.data.objects['Camera']

        if axis == "z":
            obj.rotation_euler = (0, 0, angle)
        elif axis == "y":
            obj.rotation_euler = (0, angle, 0)
        else: # x axis
            obj.rotation_euler = (angle, 0, 0)

        # Get the 2D bounding box for the image
        logging.debug("Generating 2D bounding box for labelling")
        bounding_box = camera_view_bounds_2d(C.scene, cam, obj)

        render_name = "./tmp/images/" + c_name + "/" + axis + "/renders/" + axis + "_{}-{}-{}-{}-{}.png".format(
            axis_index, bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3])

        # Render the 3D object as an image
        bpy.context.scene.render.filepath = render_name

        logging.debug("rendering pose # {} on {} axis".format(axis_index, axis))

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
            # pad axis_index to 3 places
            str_index = str(axis_index)

            if len(str_index) == 1:
                str_index = "00" + str_index
            elif len(str_index) == 2:
                str_index = "0" + str_index
            final_name = "./tmp/images/" + c_name + "/" + axis + "/renders/" + str_index + ".png"
            cv2.imwrite(final_name, cropped)
    except Exception as err:
        logging.error("def render:: {}".format(err))


# Orchestrator loads the .obj file into blender workspace, thus orchestrate runs for each class
# If the object is independent meshes, it merges the meshes into a single mesh
# The single mesh is than scaled appropriately
# Finally the mesh is rotated by theta on each axis and an image is rendered to disk
def orchestrate(three_d_obj, c_name):
    global rotation_theta
    global rotation_type
    global upper_bound
    print('Entering orchestrate',three_d_obj)
    try:
        # clear default scene
        bpy.ops.object.delete() # should delete cube

        # add object
        bpy.ops.import_scene.obj(filepath='./tmp/' + three_d_obj)

        # TODO - Determine which type of models need to be joined and which do not so we don't break this

        # flood the scene with lights
        # Directly behind the camera
        #bpy.ops.object.light_add(type='SUN', location=(7.3647, -6.9795, 4.9041))

        # Add a sun directly above the object
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 6))
        bpy.context.object.data.use_shadow = False
        bpy.context.object.data.energy = 5

        # Directly below
        bpy.ops.object.light_add(type='SUN', location=(0, 0, -6))
        bpy.context.object.rotation_euler[1] = 3.14159
        bpy.context.object.data.use_shadow = False
        bpy.context.object.data.energy = 5

        # To the left
        bpy.ops.object.light_add(type='SUN', location=(0, -6, 0))
        bpy.context.object.rotation_euler[0] = 1.5708
        bpy.context.object.data.use_shadow = False
        bpy.context.object.data.energy = 5

        # To the right
        bpy.ops.object.light_add(type='SUN', location=(0, 6, 0))
        bpy.context.object.data.use_shadow = False
        bpy.context.object.rotation_euler[0] = -1.5708
        bpy.context.object.data.energy = 5

        # behind
        bpy.ops.object.light_add(type='SUN', location=(6, 0, 0))
        bpy.context.object.rotation_euler[1] = 1.5708
        bpy.context.object.data.use_shadow = False
        bpy.context.object.data.energy = 5

        # # In front
        # bpy.ops.object.light_add(type='SUN', location=(5, 0, 0))

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
                    #bpy.ops.paint.texture_paint_toggle()

        #  Set the image background to be transparent
        C.scene.render.film_transparent = True

        #  scale object
        #s = 2/max(C.active_object.dimensions)
        s = right_size(C.active_object.dimensions)

        C.active_object.scale = (s, s, s)

        #  rotate and render
        obj = C.active_object
        obj.rotation_mode = 'XYZ'
        start_angle = 0

        # TODO https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex
        # Need to verify all vertices of the object are in the viewport

        # Rotate on the Zed axis
        print('rotation_type:'+rotation_type)
        if (rotation_type=='Z' or rotation_type=='A'):
            for z in range(1, upper_bound):
                angle = (start_angle * (math.pi/180)) + (z*-1) * (rotation_theta * (math.pi/180))
                render(obj, angle, "z", z, c_name)

        # Rotate on the X axis
        if (rotation_type=='X' or rotation_type=='A'):
            for x in range(1, upper_bound):
                angle = (start_angle * (math.pi/180)) + (x*-1) * (rotation_theta * (math.pi/180))
                render(obj, angle, "x", x, c_name)
        # Rotate on the Y axis
        if (rotation_type=='Y' or rotation_type=='A'):
            for y in range(1, upper_bound):
                angle = (start_angle * (math.pi/180)) + (y*-1) * (rotation_theta * (math.pi/180))
                render(obj, angle, "y", y, c_name)

    except Exception as err:
        logging.error("def orchestrate:: {}".format(err))
        print('Error occured',err)


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
            print('classes',classes)
            create_classes_workspace(classes)

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
    
    global rotation_theta
    global rotation_type
    global upper_bound
    rotation_type='A'
    logging.info("******************************")
    logging.info("New ODIN rendering session started at {}".format(datetime.datetime.now()))
    logging.info("******************************")
    argv = sys.argv
    argv = argv[argv.index("--")+1:] # Get all the args after "--"
    print('euler_engine_args:',argv)    
    
    rotation_theta = int(argv[0])
    print("theta set to {}".format(rotation_theta))
    s3_bucket = argv[1]
    print("s3_bucket = {}".format(s3_bucket))
    if (len(argv)>2):
        rotation_type = argv[2]
        print("rotation_type = {} (X/Y/Z/A)".format(rotation_type))
        if (len(argv)>3):
            upper_bound = int(argv[3])
            print("upper_bound = {} (0-360)".format(upper_bound))
        else:
            upper_bound=360

        
    main()
    logging.info("**************Finished execution****************")