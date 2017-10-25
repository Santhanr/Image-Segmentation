'''Source listing:
1. http://docs.opencv.org/3.2.0/d7/d4d/tutorial_py_thresholding.html
2. https://www.pyimagesearch.com/2015/11/02/watershed-opencv/
'''
import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed

# Load an color image in grayscale
img = cv2.imread('image2.jpg')
image = cv2.imread('image2.jpg')
# Convert the image to LAB color space
img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow('LAB image',img_LAB)

# Apply mean shift filter
img = cv2.pyrMeanShiftFiltering(img_LAB, 10, 20)
cv2.imshow('filtered image',img)

######## We apply Otsu's thresholding here ########

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
th, dst = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

###################################################
# Finding Contours
_,countours,hierarchy=cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print("[INFO] {} unique contours found".format(len(countours)))

# Draw Contour
cv2.drawContours(image,countours,-1,(0,255,0),3)
cv2.drawContours(gray,countours,-1,(0,255,0),3)
cv2.imshow("Contour",image)
cv2.imshow("Gray",gray)


###################################################

cv2.waitKey(0)
cv2.destroyAllWindows()
