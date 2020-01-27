import os
import argparse
import subprocess
import redis
import botocore
import boto3
import json

'''
Daemon will:
verify the presence of necessary files (Manifest + Config)
verify the 3D files within the manifest are present
load the 3D file names into a redis cache
call blender to process the files within the manifest (via the redis cache)
monitor blender progress
'''
redis_host = "localhost"
redis_port = 6379
redis_password = ""

s3_bucket = ""
s3_subfolder = ""
manifest = ""

def read_manifest():
    """
    Verify each file in the manifest is present and load a reference to each into the redis cache
    """

    global s3_bucket
    global s3_subfolder
    global manifest

    # Download the manifest file from S3
    s3 = boto3.resource('s3')
    try:
        s3.Bucket(s3_bucket).download_file(s3_subfolder + "/" + manifest, 'manifest')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The manifest file does not exist.")
        else:
            raise

    with open("manifest") as f:
        content = f.readlines()

    # Remove whitespace
    content = [x.strip() for x in content]

    if(len(content) < 1):
        print("The manifest is empty!")
        return

    try:

        # Set the Redis context
        r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
        print("writing object signatures to Redis...")
        for x in content:
            r.set("process", s3_bucket + "/" + s3_subfolder + "/" + x)

    except Exception as e:
        print("Error writing to Redis: {}".format(e))


    # Call Blender to process images
    blender = subprocess.run(["blender", "--python", "odin.py"],
                             stdout=subprocess.PIPE,
                             universal_newlines=True)

    print(blender.stdout)


def get_config_data(config_path):
    global s3_bucket
    global s3_subfolder
    global manifest

    with open(config_path) as json_file:
        config = json.load((json_file))
        s3_bucket = config['s3-bucket']
        s3_subfolder = config['s3-subfolder']
        manifest = config['manifest']

    return s3_bucket, s3_subfolder, manifest


def read_config(config_path):
    """
    verify the config file exists
    """
    # Check in current directory first
    work = os.path.join(os.getcwd(), config_path)
    if os.path.isfile(work):
        bucket, subfolder, manifest = get_config_data(work)
        # Log this -> print("Bucket = {} Subfolder = {} manifest = {}".format(bucket, subfolder, manifest))
        return True
    else:
        print("Unable to find config file!")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="The JSON config file with 3D Objects/manifest location",
                        required=True)
    args = parser.parse_args()

    if(read_config(args.config)):
        read_manifest()


if __name__ == '__main__':
    main()