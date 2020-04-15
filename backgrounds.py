# Management for background image
import cv2
import boto3
import os
import shutil
import io
from PIL import Image
import numpy as np



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
                    pass # Skip this troublemaker

    def get_background(self, index):
        return self.catalog[index]


# Debug
#generator = Generator(s3_background_bucket)
#image = cv2.cvtColor(generator.get_background(0), cv2.COLOR_RGB2BGR)
#cv2.imshow('bck', image)
#cv2.waitKey()
