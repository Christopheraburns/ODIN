

Dependencies


Python3.6 +
Blender 2.81+

	If you choose to participate in the creation of ODIN, the blender documentation is your new best friend.
	Install OpenCV in Blender python environment:
	cd <path to blender>\2.81\python\bin 
	./python3.7m -m ensurepip 
	./python.exe -m pip install --upgrade pip 
	./python.exe -m pip install opencv-python –user



Euler_Rotation Engine – The only engine that requires blender.

    • Imports the 3D model into blender.
    • Merges model components into a single object
    • Scales the model to fit the camera’s viewable area
    • Calculates the bounding box of the Image
    • Renders images of the 3D model at all degrees along the X, Y and Z axis resulting in 360*3 unique images
    • Saves .png images to the output directory (S3 bucket) with annotation metadata

Augmentation Engine 
	#TODO – need the ability to increase or decrease any of the augmentations based on feedback from the test harness

    • lighting/contrast
    • Gaussian Noise
    • “salt and pepper” (enhanced granularity)
    • region dropout (partially obscure portions of 3D model)
    weather (Fog, rain, snow)
    • Random positioning of the 3D object on the canvas (must develop normal distribution enforcer)
    • Scale the object (must develop normal distribution enforcer/tracker)
    • Matte the object on random background


Background Images
    Create an S3 bucket to store background images. Make note of the name
    Move the contents of this file: https://odin-v1.s3.amazonaws.com/backgrounds.zip into the S3 bucket - not the zip file itself, the CONTENTS of the zip file
    Grab this manifest file here: https://odin-v1.s3.amazonaws.com/manifest.txt It is a list of all the files you moved into an s3 bucket in the previous step
    Update the variable s3_background_bucket in the code to equal the name of the bucket you just created
    

Latest command line:
blender-softwaregl --background --python script.py -- [s3 bucket] [theta]

s3_bucket is where your .obj files are stored


TODO: Speed up
on Intel Core i7 avg 3.5 seconds per image.  Each class has 1080 images (before augmentation) = 63 minutes per class.