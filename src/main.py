import numpy as np
import cv2
import os

root_dir = "../img/"
filename_lists = os.listdir(root_dir)
print(filename_lists)

# img_lists = [os.path.join(root_dir, name) for name in filename_lists]
img_lists = [cv2.imread(os.path.join(root_dir, name), cv2.IMREAD_GRAYSCALE) for name in filename_lists]


# print(img_lists)
query_img = cv2.imread("../img/1.jpg",  cv2.IMREAD_GRAYSCALE)


# print(query_img.shape)
print(query_img.shape)


hist = cv2.calcHist([query_img],[0],None,[256],[0,256])
print(hist.shape)
# cv2.imshow('query', query_img)
cv2.imshow('query-gray', query_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


