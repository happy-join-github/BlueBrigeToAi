import cv2
import numpy as np
dst_size = (224,224)
Image_std = [0.229, 0.224, 0.225]
Image_mean = [0.485, 0.456, 0.406]
_std = np.array(Image_std).reshape((1, 1, 3))
_mean = np.array(Image_mean).reshape((1, 1, 3))

image_src = cv2.imread('Imagedata/n01440764/ILSVRC2012_val_00030740.JPEG')
image = cv2.resize(image_src,(256,256))
height, width = image.shape[:2]
startx, starty = width//2-dst_size[0]//2, height//2-dst_size[1]//2
image = image[startx:startx+dst_size[0],starty:starty+dst_size[1]]
normalized_image = (image - _mean) / _std

cv2.imwrite('02.jpeg',normalized_image)