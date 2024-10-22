# Each instance of the euler_engine will download all background images from S3 into memory to eliminate rapid IO
# TODO - determine if this is best way to get
# TODO - add continuation token to get all background images
# burnsca@

import cv2
import boto3
import os
import shutil
import io
from PIL import Image
import numpy as np
import random


# On start - download all backgrounds from S3
# Resize all backgrounds to desired size, i.e. 700 x 700
# load all backgrounds into memory

class Generator:
    def __init__(self, background_bucket, target_h, target_w):
        self.target_size = (target_h, target_w)
        self.catalog = []
        print("Generating in-memory database for background images")
        self.download_backgrounds(background_bucket, self.catalog)

    def download_backgrounds(self, bucket, catalog):
        if os.path.exists('./tmp/backgrounds'):
            # delete any existing files in background folder
            shutil.rmtree('./tmp/backgrounds')
        else:
            # create background folder
            os.mkdir('./tmp/backgrounds')

        s3 = boto3.client('s3')
        list = s3.list_objects(Bucket=bucket)['Contents']
        # TODO - this only grabs the first 1000 entries.  Need to put an iterator in here.
        index = 1
        for key in list:
            if key['Key'] != '.directory':
                try:
                    print("Loaded {} images into background catalog".format(index))
                    img_stream = io.BytesIO()
                    s3.download_fileobj(bucket, key['Key'], img_stream)
                    # PIL reads from bytestream nicely - read in PIL.Image format
                    img = Image.open(img_stream)
                    # Resize to our target size
                    resized = img.resize(self.target_size)
                    # Convert to OpenCV image for storing in catalog
                    cv_image = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)
                    self.catalog.append(cv_image)
                    index += 1
                except Exception as err:
                    # TODO - NEED to log this error somewhere
                    print("unable to add {}.  Error = {}".format(key['Key'], err))
                    pass # Skip this troublemaker

    def get_background(self, index):
        return self.catalog[index]

    def get_count(self):
        return len(self.catalog)

# Debug
#generator = Generator('odin-bck', 700, 700)

#index = random.randrange(0, generator.get_count())
#image = cv2.cvtColor(generator.get_background(index), cv2.COLOR_RGB2BGR)
#cv2.imshow(index, image)
#cv2.waitKey()
