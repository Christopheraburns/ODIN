# Large datasets are unwieldy. Use this to begin building tools to manage dataset debugging/resetting/etc.

import os
import shutil

path = './tmp/images'
axis = ['x', 'y', 'z']
classes = ['ballpean', 'boxwrench', 'crafthammer', 'framinghammer', 'mallet']

shutil.rmtree('./tmp/annotations')
os.mkdir('./tmp/annotations')

for c in classes:
    for ax in axis:
        workpath = path + '/' + c + '/' + ax + '/ss'
        if os.path.exists(workpath):
            shutil.rmtree(workpath)
            os.mkdir(workpath)