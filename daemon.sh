#!/bin/bash
#!/usr/bin python3

echo "Blender is rendering images..."

# Use blender to render the image on each degree of X , Y and Z axis then exit
blender -b -noaudio -E CYCLES --python euler_engine.py -- handtools 1

echo "Blender is finished rendering, begin augmentation process..."

#python3 augment_engine.py