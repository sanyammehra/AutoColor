import glob
import numpy as np
from PIL import Image


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

i = 1

for filename in glob.iglob('/Users/kushaagragoyal/Desktop/CS231n/Project/tiny-imagenet-200/train/**/**/*.JPEG',recursive=True):
    print(filename)
    img = Image.open(filename).convert('LA')
    img.save('grayscale/train/' + str(i) + '.png')
    i += 1
    


