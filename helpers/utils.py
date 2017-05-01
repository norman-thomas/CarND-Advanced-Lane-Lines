import cv2
import numpy as np

def load_image(fname, gray=False, debug=False):
    img = np.array(cv2.imread(fname))
    if debug:
        print('Opened file {}, with shape {}'.format(fname, img.shape))

    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
