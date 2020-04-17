# Large datasets are unwieldy. Use this to begin building tools to manage dataset debugging/resetting/etc.
# burnsca@
import numpy as np
import os
import shutil
import cv2

path = './tmp/images'
axis = ['x', 'y', 'z']
classes = ['ballpean', 'boxwrench', 'crafthammer', 'framinghammer', 'mallet']
subs = ['renders', 'base', 'ss']


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


def prepare_review_frames():
    # Create a movie of all the images rendered to review for quality
    if os.path.exists('./tmp/review'):
        shutil.rmtree('./tmp/review')

    os.mkdir('./tmp/review')
    master_count = 1
    for c in classes:
        for a in axis:
            workdir = path + '/' + c + '/' + a + '/renders'
            file_list = []
            # Move images to temp folder and make sure all are sized the same
            for root, dir, files in os.walk(workdir):
                for file in files:
                    # Create a list to be sorted
                    file_list.append(file)

            # Sort and Iterate the list we just created
            file_list.sort()
            r = range(0, len(file_list))
            for x in r:
                # Create a transparent background
                background = np.zeros((1080, 1920, 3), dtype=np.uint8)
                # read the rendered image from disk
                render = cv2.imread(workdir + '/' + file_list[x])
                # overlay the render on our transparent background
                x = 910
                y = 540

                # find center point of rendered image
                xx = int(render.shape[1] / 2)
                yy = int(render.shape[0] / 2)

                # Subtract center point (X,Y) of rendered image from the center point (X, Y) of the background image
                # to position the rendered image in the center of the background
                x_min = int(round(x - xx))
                y_min = int(round(y - yy))

                # Merge the background and rendered image
                final_img = overlay_transparent(background, render, x_min, y_min)

                str_index = str(master_count)
                if len(str_index) == 1:
                    str_index = "000" + str_index
                elif len(str_index) == 2:
                    str_index = "00" + str_index
                elif len(str_index) == 3:
                    str_index = "0" + str_index

                cv2.imwrite('./tmp/review/' + str_index + '.png', final_img)
                master_count += 1


def render_review():
    try:
        os.system('ffmpeg -framerate 24 -i ./tmp/review/%04d.png review.mp4')

        # Clean up and delete ./tmp/review
        shutil.rmtree('./tmp/review')
    except Exception as err:
        print(err)


def create_review():
    prepare_review_frames()
    render_review()


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


#clear_ss()
#clear_base()
#clear_annotations()

#spiral_work()

render_review()