################
#blender script#
################
##to run in the scripting window

import math
from mathutils import *
import bpy

def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.

    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

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
        return (0, 0, 0, 0)

    return (
        round(min_x * dim_x),            # X
        round(dim_y - max_y * dim_y),    # Y
        round((max_x - min_x) * dim_x),  # Width
        round((max_y - min_y) * dim_y)   # Height
    )

try:
    
    C = bpy.context
    output_folder="/home/chris/Desktop/blender"
    output_filename="obj_out"
    logfile="/home/chris/Desktop/logfile.txt"

    #clear default scene
    bpy.ops.object.delete() # should delete cube

    #add object
    file_loc = "/home/chris/Downloads/cams650.obj"
    bpy.ops.import_scene.obj(filepath = file_loc)

    #We don't need to join objects in Blender 2.8x
    #TODO - Determine which type of models need to be joined and which do not so we don't break this
    for ob in bpy.context.scene.objects:
        if ob.type=="MESH":
            ob.select_set(True)
            # TODO - this makes the assumption there is only one mesh  - could end badly
            bpy.context.view_layer.objects.active = ob
        else:
            ob.select_set(False)
    bpy.ops.object.join()

    # transparent sky
    C.scene.render.film_transparent = True

    #scale object
    k=2/max(C.active_object.dimensions)

    C.active_object.scale=(k,k,k)

    upper = 10
    #rotate and render

    obj = C.active_object
    obj.rotation_mode='XYZ'
    cam = bpy.data.objects['Camera']
    theta=1 #degrees to turn per rotation
    start_angle=0

    # Rotate on the Z axis
    for z in range(1,upper):
        angle = (start_angle * (math.pi /180)) + (z*-1) * (theta * (math.pi/180))
        obj.rotation_euler  = (0,0,angle)
        filename = output_folder + "/" + output_filename + "_z_{}.png".format(z)
        bpy.context.scene.render.filepath = filename
        bpy.ops.render.render(write_still=True,use_viewport=True)

        with open(output_folder + "/z_bounds_" + str(z) + ".txt", "w") as f:
            f.write(camera_view_bounds_2d(C.scene, cam, obj) + "\n")

        '''
        z_vertices=[] 
        for c in obj.bound_box:
           if [Vector(c).x,Vector(c).y,Vector(c).z] not in z_vertices:
                z_vertices.append([Vector(c).x,Vector(c).y,Vector(c).z])

        with open(output_folder +"/z_bounds_" + str(z) + ".txt","w") as f:
            for zv in z_vertices:
                f.write(str(zv[0]) + "," + str(zv[1]) + "," + str(zv[2]) + "\n")
        '''

    for x in range(1,upper):
        angle = (start_angle * (math.pi /180)) + (x*-1) * (theta * (math.pi/180))
        obj.rotation_euler  = (angle,0,0)
        bpy.context.scene.render.filepath  =output_folder + "/" + output_filename + "_x_%d.png" % (x)
        bpy.ops.render.render(write_still=True,use_viewport=True)

        with open(output_folder + "/x_bounds_" + str(x) + ".txt", "w") as f:
            f.write(camera_view_bounds_2d(C.scene, cam, obj) + "\n")

        '''
        x_vertices=[]
        for c in obj.bound_box:
           if [Vector(c).x,Vector(c).y,Vector(c).z] not in x_vertices:
                x_vertices.append([Vector(c).x,Vector(c).y,Vector(c).z])

        with open(output_folder +"/x_bounds_" + str(x) + ".txt","w") as f:
            for zv in x_vertices:
                f.write(str(zv[0]) + "," + str(zv[1]) + "," + str(zv[2]) + "\n")
        '''

    for y in range(1,upper):
        angle = (start_angle * (math.pi /180)) + (y*-1) * (theta * (math.pi/180))
        obj.rotation_euler  = (0,angle,0)
        bpy.context.scene.render.filepath  =output_folder + "/" + output_filename + "_y_%d.png" % (y)
        bpy.ops.render.render(write_still=True,use_viewport=True)

        with open(output_folder + "/y_bounds_" + str(y) + ".txt", "w") as f:
            f.write(camera_view_bounds_2d(C.scene, cam, obj) + "\n")

        '''
        y_vertices=[]
        for c in obj.bound_box:
           if [Vector(c).x,Vector(c).y,Vector(c).z] not in y_vertices:
                y_vertices.append([Vector(c).x,Vector(c).y,Vector(c).z])
        with open(output_folder +"/y_bounds_" + str(y) + ".txt","w") as f:
            for zv in y_vertices:
                f.write(str(zv[0]) + "," + str(zv[1]) + "," + str(zv[2]) + "\n")
        '''

except Exception as msg:
    with open(logfile,"w") as f:
        f.write(str(msg))
