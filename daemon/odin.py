import sys
import os
import boto3
import redis
import bpy
import mathutils

# Set up redis cache
#TODO - should change from redis to Dynamo to persist unprocessed images in the event the Blender AMI exits
redis_host = "localhost"
redis_port = 6379
redis_password = ""

def get_scalefactor(x,y,z):
    # The largest dimension should be no larger than 17
    scale_x = 17 / x
    scale_y = 17 / y
    scale_z = 17 / z

    # return the smallest scale factor
    if scale_x > scale_y:
        if scale_y < scale_z:
            return scale_y
        else:
            return scale_z
    else:
        if scale_x > scale_z:
            return scale_z
        else:
            return scale_x


# Remove all the default assets from the scene
def clear_scene():
   # remove mesh Cube
   if "Cube" in bpy.data.meshes:
       mesh = bpy.data.meshes["Cube"]
       print("removing mesh", mesh)
       bpy.data.meshes.remove(mesh)

   # remove the default Lamp
   bpy.data.objects['Lamp'].select = True
   print("Deleting default light")
   bpy.ops.object.delete()

   # remove the default camera
   bpy.data.objects['Camera'].select = True
   print("Deleting default camera")
   bpy.ops.object.delete()

   return True


def join_meshes(scene):
   if len(bpy.data.objects) > 1:
      obs = []
      for ob in scene.objects:
         if ob.type == 'MESH':
            obs.append(ob)

      ctx = bpy.context.copy()
      # make one object "active"
      ctx['active_object'] = obs[0]
      ctx['selected_objects'] = obs

      bpy.ops.object.join(ctx)


def scale_model(x, y, z, scene):
    if x > 17 or y > 17 or z > 17:
        # Scale Down
        ctx = bpy.context.copy()
        ctx['active_object'] = scene.objects[0]
        scale_factor = get_scalefactor(x,y,z)
        bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor), constraint_axis=(False, False, False),
                                 constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',
                                 proportional_edit_falloff='SMOOTH', proportional_size=1)


def add_tracking_camera():

    # Add a new camera
    bpy.ops.object.camera_add(view_align=True, enter_editmode=False, location=(0, 14.7996, 11.9692),
                              rotation=(0.700999, 0.0100532, -3.07246),
                              layers=(True, False, False, False, False, False, False, False, False, False, False, False,
                              False, False,False, False, False, False, False, False))

    # Get a reference to the newly added camera
    #TODO - how can we get a context reference to the camera when we create it?
    cam = bpy.context.object
    if cam.type != 'CAMERA':
        print("Unable to get a reference to the camera!")
        return False

    # Add a TRACK_TO constraint to the Camera
    constraint = cam.constraints.new('TRACK_TO')
    cons_id = constraint.name

    model = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            model = obj

    cam.constraints[cons_id].target = model
    cam.constraints[cons_id].up_axis = 'UP_Y'

    return True


def preprocess_model(file):
    success = True

    # get a scene context
    scene = bpy.context.scene

    # Clear the scene of all default assets
    if not clear_scene():
        success = False

    # import the 3D object
    bpy.ops.import_scene.obj(filepath=file)
    #TODO - verify that the import was successful

    # If the imported model has more than one mesh - join the meshes
    join_meshes(scene)

    # Get reference to the singular mesh
    model = bpy.data.objects[0]

    # We should now have a single Object within the scene - let's get its dimensions
    x, y, z = model.dimensions

    # Scale the model down if it is too large
    scale_model(x, y, z, scene)

    # Add a tracking camera to render images
    add_tracking_camera()


    '''
    bpy.ops.render.opengl(animation=False, sequencer=False,write_still=True, view_context=True)

    bpy.data.images['Render Result'].save_render(filepath="scripted.png")
    '''

    #TODO - Implement the rotation-export function (with bounding boxes)
    #TODO - Implement the imgaug libary based on config variables to add random backgrounds
    #TODO - Export the image to S3 with VOC data
    #TODO - Once the render is complete and verified - delete the entry from the Redis Cache


def main():
    print("loading blender")
    obj_list = []
    s3 = boto3.resource('s3')

    print("Querying redis cache for unprocessed images...")
    # Get all the objects to process from the redis cache
    r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
    for key in r.scan_iter("process:*"):
        obj = r.get(key)
        obj_list.append(obj)


    #TODO - this syntax assumes all instances will have 1 Bucket, 1 Sub_folder and then the files.  It will break with
    # more than one or less than one sub-folder
    for obj in obj_list:

        s3_bucket = obj[:obj.index("/")]
        key = obj[obj.index("/") +1 :]
        fname = key[key.index("/") +1 :]
        print("downloading {} to local disk for processing".format(fname))
        if os.path.exists(fname):
            os.remove(fname)

        s3.Bucket(s3_bucket).download_file(key, fname)
        print("ODIN downloaded file: {}".format(fname))
        preprocess_model(fname)


if __name__ == '__main__':
    main()