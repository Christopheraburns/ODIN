# Large datasets are unwieldy. Use this to begin building tools to manage dataset debugging/resetting/etc.
# burnsca@

import os
import shutil

path = './tmp/images'
axis = ['x', 'y', 'z']
classes = ['ballpean', 'boxwrench', 'crafthammer', 'framinghammer', 'mallet']
subs = ['renders', 'base', 'ss']


def clear_annotations():
    shutil.rmtree('./tmp/annotations')
    os.mkdir('./tmp/annotations')



def clear_base():
    for c in classes:
        for ax in axis:
            workpath = path + '/' + c + '/' + ax + '/base'
            if os.path.exists(workpath):
                shutil.rmtree(workpath)
                os.mkdir(workpath)

def clear_ss():
    for c in classes:
        for ax in axis:
            workpath = path + '/' + c + '/' + ax + '/ss'
            if os.path.exists(workpath):
                shutil.rmtree(workpath)
                os.mkdir(workpath)


def create_classes_workspace(path=path, classes=classes, subs=subs):
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
        print(err)
        #logging.error("def create_workspace_classes:: {}".format(err))


#create_classes_workspace()


def spiral_work():
    spiral_trajectory_points = 90
    for p in range(spiral_trajectory_points + 1):
        if p > 0 and p % 30 == 0:
            print("Hit: P={}".format(p))


clear_ss()
clear_base()
clear_annotations()


#spiral_work()