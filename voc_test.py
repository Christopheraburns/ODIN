import os, zipfile
from gluoncv import utils
import mxnet as mx
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET

img_filename = "skidsteer_z_1.png"
annot_filename = "skidsteer_z_1.xml"
class_name = "skidsteer"
img = mx.image.imread(img_filename)

bounding_box = [885, 213, 316, 515]


'''
ax = utils.viz.plot_image(img)
print(img.shape)
plt.show()

label = [885, 213, 1201, 728]
box = np.array([label])
ids = np.array([0])
class_names = ['skidsteer']

ax = utils.viz.plot_bbox(img, box, labels=ids, class_names=class_names)
plt.show()
'''

# Build the Structure of the VOC File
annot = ET.Element('annotation')
fname = ET.SubElement(annot, 'filename')
size = ET.SubElement(annot, 'size')
img_width = ET.SubElement(size, 'width')
img_height = ET.SubElement(size, 'height')
img_depth  = ET.SubElement(size, 'depth')
obj_node = ET.SubElement(annot, 'object')
class_name_node = ET.SubElement(obj_node, 'name')
diff = ET.SubElement(obj_node, 'difficult')
bndbox = ET.SubElement(obj_node, 'bndbox')
xmin_node = ET.SubElement(bndbox, 'xmin')
ymin_node = ET.SubElement(bndbox, 'ymin')
xmax_node = ET.SubElement(bndbox, 'xmax')
ymax_node = ET.SubElement(bndbox, 'ymax')

fname.text = annot_filename
# TODO - OpenV will add a random background, crop the image to a more manageable size and save the size in a variable
img_width.text = str(img.shape[1])
img_height.text = str(img.shape[0])
img_depth.text = str(img.shape[2])

class_name_node.text = class_name
diff.text = "0" # TODO - what does this even mean?
xmin_node.text = str(bounding_box[0])
ymin_node.text = str(bounding_box[1])
xmax_node.text = str(bounding_box[0] + bounding_box[2])
ymax_node.text = str(bounding_box[1] + bounding_box[3])


xml_str = ET.tostring(annot).decode()

annot_file = open(annot_filename, "w")
annot_file.write(xml_str)