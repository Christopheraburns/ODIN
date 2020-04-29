# Module to shuffle and sort completed images into a VOC dataset. (Annotation files are already written)
# burnsca@

import os
import random
import datetime
import shutil
import sys
import boto3

s3 = boto3.client('s3')

classes = []
path = './tmp/images'


# Create list of files in each target folder
# shuffle list
# move 70% of images to intermediate training directory
# move annotation files to S3 Staging directory VOCTrain
# move remaining images to intermediate valid directory
# move remaining annotaion file to S3 Staging directory VOCValid


# move dataset to S3

# Returns a clean list of files within target directory
def generate_list(workpath):
    file_list = []
    for root, dir, files in os.walk(workpath):
        for file in files:
            file_name, ext = os.path.splitext(file)
            if ext == '.jpg':
                file_list.append(file)

    return file_list


def split(c_name, axis, subfolder):
    global job_id

    try:
        # Capture some paths in variables
        img_source = path + '/' + c_name + '/' + axis + '/' + subfolder + '/'
        if os.path.exists(img_source):
            print('img_source found')
            annot_source = './tmp/annotations/'

            voctrain_img_dest = './tmp/' + job_id + '/VOCTrain/shuffle/'
            vocvalid_img_dest = './tmp/' + job_id + '/VOCValid/shuffle/'

            voctrain_annot_dest = './tmp/' + job_id + '/VOCTrain/Annotations/'
            vocvalid_annot_dest = './tmp/' + job_id + '/VOCValid/Annotations/'


            # Create list of all images in working path
            file_list = generate_list(img_source)

            # Shuffle the list
            random.shuffle(file_list)

            # Get 70% for training
            seventy = round(len(file_list) * .7)

            # Copy the 70% VOC structure (training)
            sev = range(0, seventy)
            for i in sev:
                file = file_list[i]
                # Copy Image file to VOCTrain staging
                shutil.copyfile(img_source + file, voctrain_img_dest + file)

                # Copy Annot file to VOCTrain
                file_name, ext = os.path.splitext(file)
                shutil.copyfile(annot_source + file_name + '.xml', voctrain_annot_dest + file_name + '.xml')


            # Copy the 30% to VOC structure (validation)
            thi = range(seventy + 1, len(file_list))
            for i in thi:
                file = file_list[i]
                # Copy Image file to VOCValid staging
                shutil.copyfile(img_source + file, vocvalid_img_dest + file)

                # Copy Annot file to VOCValid
                file_name, ext = os.path.splitext(file)
                shutil.copyfile(annot_source + file_name + '.xml', vocvalid_annot_dest + file_name + '.xml')
    except Exception as err:
        print(err)


def create_voc_directory():
    global job_id

    # Directory Structure for VOC
    os.mkdir('./tmp/' + job_id)
    os.mkdir('./tmp/' + job_id + '/VOCTrain')
    os.mkdir('./tmp/' + job_id + '/VOCTrain/shuffle')
    os.mkdir('./tmp/' + job_id + '/VOCTrain/JPEGImages')
    os.mkdir('./tmp/' + job_id + '/VOCTrain/Annotations')
    os.mkdir('./tmp/' + job_id + '/VOCTrain/ImageSets')
    os.mkdir('./tmp/' + job_id + '/VOCTrain/ImageSets/Main')
    os.mkdir('./tmp/' + job_id + '/VOCValid')
    os.mkdir('./tmp/' + job_id + '/VOCValid/shuffle')
    os.mkdir('./tmp/' + job_id + '/VOCValid/JPEGImages')
    os.mkdir('./tmp/' + job_id + '/VOCValid/Annotations')
    os.mkdir('./tmp/' + job_id + '/VOCValid/ImageSets')
    os.mkdir('./tmp/' + job_id + '/VOCValid/ImageSets/Main')

def move_to_s3():
    
    global bucket
    global job_id

    print('move_to_s3')

    for root, dirs, files in os.walk('./tmp/' + job_id):
        print('files',files)
        for filename in files:
            path=os.path.join(root, filename)
            key= path.replace('./tmp/','synthetic_data/')
            print('uploading',path)
            upload_file(path, bucket, object_name=key)

def upload_file(file_name, bucket, object_name=None):

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    try:
        response = s3.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

# Orchestrate the folder enumeration and split each class/axis/image type
def split_data():
    global job_id

    axis = ['x', 'y', 'z']
    subs = ['base', 'ss']
    for c_name in classes:
        for a in axis:
            for s in subs:
                print("********************************")
                print("splitting {} data for {} on {} axis".format(s, c_name, a))
                print("********************************")
                split(c_name, a, s)


def build_VOC():
    voctrain_set_dest = './tmp/' + job_id + '/VOCTrain/ImageSets/Main/train.txt'
    vocvalid_set_dest = './tmp/' + job_id + '/VOCValid/ImageSets/Main/valid.txt'

    # Shuffle all training data
    train_list = generate_list('./tmp/' + job_id + '/VOCTrain/shuffle/')

    #Shuffle all validation data
    valid_list = generate_list('./tmp/' + job_id + '/VOCValid/shuffle/')

    # Shuffle the lists
    random.shuffle(train_list)
    random.shuffle(valid_list)

    # Move the shuffled training files out of staging and write the ImageSet file
    for t in range(len(train_list)):
        print('moving training file',train_list[t])

        # Move the file
        file = train_list[t]
        file_name, _ = os.path.splitext(file)
        img_source = './tmp/' + job_id + '/VOCTrain/shuffle/'
        img_dest = './tmp/' + job_id + '/VOCTrain/JPEGImages/'
        shutil.copyfile(img_source + file, img_dest + file)
        # Delete the file after copying to avoid too much disk usage
        #os.remove(img_source + file)

        # Write filename to train.txt
        with open(voctrain_set_dest, 'a+') as f:
            f.write(file_name + "\n")

    # Validation data
    for v in range(len(valid_list)):
        # Move the file
        file = valid_list[v]
        file_name, _ = os.path.splitext(file)
        img_source = './tmp/' + job_id + '/VOCValid/shuffle/'
        img_dest = './tmp/' + job_id + '/VOCValid/JPEGImages/'
        shutil.copyfile(img_source + file, img_dest + file)
        # Delete the file after copying to avoid too much disk usage
        os.remove(img_source + file)

        # Write filename to train.txt
        with open(vocvalid_set_dest, 'a+') as f:
            f.write(file_name + "\n")


    # Delete the staging folders from VOC structure
    shutil.rmtree('./tmp/' + job_id + '/VOCTrain/shuffle')
    shutil.rmtree('./tmp/' + job_id + '/VOCValid/shuffle')

def main():
    # Create the VOC structure
    create_voc_directory()

    # Split the data into training and validation
    split_data()

    # Move to VOC folder structure, write ImageSet lists
    build_VOC()

    # Move to S3
    move_to_s3()


if __name__ == '__main__':
    
    global job_id
    global bucket

    if (len(sys.argv)>0):
        bucket = sys.argv[-1]
        print('bucket',bucket)    
        if (len(sys.argv)>1):
            job_id = sys.argv[-2]
        else:
            job_id = '2020-04-0712-27-23-324283'
        print('job_id',job_id)    

    for root, dirs, files in os.walk('./tmp'):
        # only process the .obj files for now
        for filename in files:
            class_name, ext = os.path.splitext(filename)
            if ext.lower() == '.obj':
                classes.append(class_name)
    main()




