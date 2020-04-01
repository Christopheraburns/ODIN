# Import modules
import json
import re
from flask import Flask
from flask import request
from joblib import dump, load
import numpy as np
import json
import os
app = Flask(__name__)

# Create a path for health checks
@app.route("/ping")
def endpoint_ping():
    return ""



# Create a path for inference
@app.route("/invocations", methods=["POST"])
def endpoint_invocations():
    payload = json.loads(request.get_data().decode("utf8"))
    print(payload['bucket_name'])
    print(payload['s3_prefix_three_d_object'])
    print(payload['s3_prefix_output'])
    print(payload['s3_prefix_backgrounds'])
    print(payload['theta'])
    command='blender-2.82a-linux64/blender -b -noaudio -E CYCLES --python script3.py '+payload['bucket_name'] +" "+payload['s3_prefix_three_d_object']+" "+payload['s3_prefix_backgrounds']+" "+payload['theta']+" "+payload['s3_prefix_output']
    print('running command:'+command)
    stream = os.popen(command)
    output = stream.read()
    output
    #os.system(command)
    return output