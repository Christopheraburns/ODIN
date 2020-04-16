# Module to shuffle and sort completed images into a VOC dataset. (Annotation files are already written)
# burnsca@

import os
import random
import datetime
import shutil

job_id = '2020-04-0712-27-23-324283'


def split(c_name, axis):
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
                    seventy = round(len(image_list) * .7)

                    # Get 30% for validation
                    thirty = len(image_list) - seventy

                    # Copy the 70% VOC structure (training)
                    sev = range(0, seventy)
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

                    # Copy the 30% to VOC structure (validation)
                    thi = range(seventy + 1, len(image_list))
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



classes = ['hammer', 'screwdriver2', 'wrench']



#split('hammer', 'x')

for c_name in classes:
    # print("********************************")
    # print("splitting data for {} on {} axis".format(c_name, 'x'))
    # print("********************************")
    split(c_name, 'x')
    # print("********************************")
    # print("splitting data for {} on {} axis".format(c_name, 'y'))
    # print("********************************")
    split(c_name, 'y')
    # print("********************************")
    # print("splitting data for {} on {} axis".format(c_name, 'z'))
    # print("********************************")
    split(c_name, 'z')


