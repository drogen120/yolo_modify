from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

img = Image.open('./vis/output_layer_1.jpg')
#img.thumbnail((7, 7))
imgplot = plt.imshow(img)
plt.show()