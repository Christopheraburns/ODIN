

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


Background Image Server
    Create an appropriately named S3 bucket to store background images
    Move the contents of this file: https://odin-v1.s3.amazonaws.com/backgrounds.zip into the S3 bucket - not the zip file itself, the CONTENTS of the zip file
    Grab this manifest file here: https://odin-v1.s3.amazonaws.com/manifest.txt It is a list of all the files you moved into an s3 bucket in the previous step
    Create a lambda function based on *this* code
    The Lambda function, when called will go grab a random image from the background image bucket and return that image to the caller