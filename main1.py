import cv2
import numpy as np
# from matplotlib.pyplot import imshow
# from google.colab.patches import cv2_imshow
# !wget  https://i.stack.imgur.com/sDQLM.png
#read image 
image = cv2.imread( "croped.png")

width = 350
height = 450
dim = (width, height)

image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

#convert to gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#performing binary thresholding
kernel_size = 3
ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)  

#finding contours 
cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

#drawing Contours
radius =2
color = (30,255,50)
cv2.drawContours(image, cnts, -1,color , radius)
# cv2.imshow(image) commented as colab don't support cv2.imshow()
cv2.imshow("",image)
# cv2.waitKey()
cv2.waitKey(0)