

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



