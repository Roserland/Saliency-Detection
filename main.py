from sklearn.cluster import KMeans
import numpy as np
import PIL
import sklearn
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from collections import Counter
import skimage
from skimage import io, color
from skimage import filters
from skimage import io, color

from functionFile import *

fileName = '6.jpg'

# fileFolderPath = 'D:\\DIP_Practice\\(ASD)Image+GT'
# fileName = fileFolderPath + '\\' + '110.jpg'

rgb = io.imread(fileName)
lab = color.rgb2lab(rgb)
hsv = color.rgb2hsv(rgb)

rgb_res = getClusterResult(rgb)
lab_res = getClusterResult(lab)
hsv_res = getClusterResult(hsv)

res_vote = voter(lab_res, rgb_res, hsv_res)

# for i in range(1):
#     res_vote = modeSmoothing(res_vote)
show2(res_vote)