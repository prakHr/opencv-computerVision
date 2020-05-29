#####################################
#                                                                      #
#        Blurring(using an Average kernel)  ######
#####################################


import os
import cv2
import numpy as np
path='C:/Users/HP/Desktop/Image'
images_files=os.listdir(path)
'''
img = cv2.imread(path+'/'+images_files[0])
rows, cols = img.shape[:2]
kernel_identity = np.array([[0,0,0], [0,1,0], [0,0,0]])
kernel_3x3 = np.ones((3,3), np.float32) / 9.0
kernel_5x5 = np.ones((5,5), np.float32) / 25.0
cv2.imshow('Original', img)
output = cv2.filter2D(img, -1, kernel_identity)
cv2.imshow('Identity filter', output)
output = cv2.filter2D(img, -1, kernel_3x3)
cv2.imshow('3x3 filter', output)
output = cv2.filter2D(img, -1, kernel_5x5)
cv2.imshow('5x5 filter', output)
#output = cv2.blur(img, (3,3))
cv2.waitKey(0)
'''
#####################################
#                                                                      #
#                                   Edge Detection  ######
#####################################


'''
img = cv2.imread(path+'/'+images_files[0], cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
#laplacian = cv2.Laplacian(img, cv2.CV_64F)
cv2.imshow('Original', img)
cv2.imshow('Sobel horizontal', sobel_horizontal)
cv2.imshow('Sobel vertical', sobel_vertical)
cv2.waitKey(0)
#Motion Blur
import cv2
import numpy as np
img = cv2.imread('input.jpg')
cv2.imshow('Original', img)
size = 15
# generating the kernel
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size
# applying the kernel to the input image
output = cv2.filter2D(img, -1, kernel_motion_blur)
cv2.imshow('Motion Blur', output)
cv2.waitKey(0)
'''

#####################################
#                                                                      #
#                                   Sharpening        ######
#####################################


'''
img = cv2.imread(path+'/'+images_files[0])
cv2.imshow('Original', img)
# generating the kernels
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
kernel_sharpen_2 = np.array([[1,1,1], [1,-7,1], [1,1,1]])
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],
[-1,2,2,2,-1],
[-1,2,8,2,-1],
[-1,2,2,2,-1],
[-1,-1,-1,-1,-1]]) / 8.0
# applying different kernels to the input image
output_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
output_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
output_3 = cv2.filter2D(img, -1, kernel_sharpen_3)
cv2.imshow('Sharpening', output_1)
cv2.imshow('Excessive Sharpening', output_2)
cv2.imshow('Edge Enhancement', output_3)
cv2.waitKey(0)
'''
#####################################
#                                                                      #
#                                   Embossing         ######
#####################################

'''
img_emboss_input = cv2.imread(path+'/'+images_files[0])
# generating the kernels
kernel_emboss_1 = np.array([[0,-1,-1],
[1,0,-1],
[1,1,0]])
kernel_emboss_2 = np.array([[-1,-1,0],
[-1,0,1],
[0,1,1]])
kernel_emboss_3 = np.array([[1,0,0],
[0,0,0],
[0,0,-1]])
# converting the image to grayscale
gray_img = cv2.cvtColor(img_emboss_input,cv2.COLOR_BGR2GRAY)
# applying the kernels to the grayscale image and adding the offset
output_1 = cv2.filter2D(gray_img, -1, kernel_emboss_1) + 128
output_2 = cv2.filter2D(gray_img, -1, kernel_emboss_2) + 128
output_3 = cv2.filter2D(gray_img, -1, kernel_emboss_3) + 128
cv2.imshow('Input', img_emboss_input)
cv2.imshow('Embossing - South West', output_1)
cv2.imshow('Embossing - South East', output_2)
cv2.imshow('Embossing - North West', output_3)
cv2.waitKey(0)
'''

#####################################
#                                                                      #
#                     Erosion and dilation         ######
#####################################


'''
img = cv2.imread(path+'/'+images_files[0],0)
kernel = np.ones((5,5), np.uint8)
img_erosion = cv2.erode(img, kernel, iterations=1)#iterations applies the operation successively
img_dilation = cv2.dilate(img, kernel, iterations=1)
cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)
cv2.waitKey(0)
'''
#####################################
#                                                                      #
#           Creating a vignetter filter            ######
#####################################


'''
img = cv2.imread(path+'/'+images_files[0])
rows, cols = img.shape[:2]
# generating vignette mask using Gaussian kernels
kernel_x = cv2.getGaussianKernel(cols,2000)#controls the radius of the bright central region
kernel_y = cv2.getGaussianKernel(rows,2000)
kernel = kernel_y * kernel_x.T
mask = 255 * kernel / np.linalg.norm(kernel)
output = np.copy(img)
# applying the mask to each channel in the input image
for i in range(3):
    output[:,:,i] = output[:,:,i] * mask
cv2.imshow('Original', img)
cv2.imshow('Vignette', output)
cv2.waitKey(0)
'''
###########################################
#                                                                                   #
#   Creating a vignetter filter by shifting focus        ######
###########################################


'''
img = cv2.imread(path+'/'+images_files[0])
rows, cols = img.shape[:2]
# generating vignette mask using Gaussian kernels
kernel_x = cv2.getGaussianKernel(int(1.5*cols),2000)
kernel_y = cv2.getGaussianKernel(int(1.5*rows),2000)
kernel = kernel_y * kernel_x.T
mask = 255 * kernel / np.linalg.norm(kernel)
mask = mask[int(0.5*rows):, int(0.5*cols):]
output = np.copy(img)
# applying the mask to each channel in the input image
for i in range(3):
    output[:,:,i] = output[:,:,i] * mask
cv2.imshow('Input', img)
cv2.imshow('Vignette with shifted focus', output)
cv2.waitKey(0)
'''
#####################################
#                                                                      #
#Enhancing the contrast in an image         ######
#####################################


'''
img = cv2.imread(path+'/'+images_files[0], 0)
# equalize the histogram of the input image
histeq = cv2.equalizeHist(img)
cv2.imshow('Input', img)
cv2.imshow('Histogram equalized', histeq)
cv2.waitKey(0)
'''
##########################################
#                                                                                #
#   Histogram equalization for color images        ######
##########################################


'''
img = cv2.imread(path+'/'+images_files[0])
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
cv2.imshow('Color input image', img)
cv2.imshow('Histogram equalized', img_output)
cv2.waitKey(0)
'''











