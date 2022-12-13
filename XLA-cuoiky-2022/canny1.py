import numpy as np
import cv2
  
img = cv2.imread("../final2.jpg", 0)  # Read image


def canny_edge_detection(image_path, blur_ksize=5, threshold1=100, threshold2=200):
    """
    image_path: link to image
    blur_ksize: Gaussian kernel size
    threshold1: min threshold 
    threshold2: max threshold
    """
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray,(blur_ksize,blur_ksize),0)

    img_canny = cv2.Canny(img_gaussian,threshold1,threshold2)

    return img_canny
    
    
image_path = "../final2.jpg"
# gray = cv2.imread(image_path, 0)

img_canny = canny_edge_detection(image_path, 25, 50, 100)
  

  
# cv2.imshow('original', img)
# cv2.imshow('edge', edge)
cv2.imshow('edge', img_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()