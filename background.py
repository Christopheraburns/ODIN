import cv2
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import random
from glob import glob
from PIL import Image





class Backgrounds():
    def __init__(self, bckgrnd="/data/backgrounds.pck"):
        self._images = pickle.load(open(bckgrnd, 'rb'))
        self._nb_images = len(self._images)

    def get_random(self, display=False):
        bg=self._images[random.randint(0, self._nb_images - 1)]
        if display:
            plt.imshow(bg)
            plt.show()
        return bg


backgrounds = Backgrounds()


def get_background():
    return backgrounds.get_random()
