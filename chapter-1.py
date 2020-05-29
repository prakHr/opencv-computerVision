#Code from book ::- Joshi P. - OpenCV with Python By Example_ Build real-world computer vision applications and develop cool demos using OpenCV for Python.pdf
#Creating the conda environment and installing libraries
'''
conda create -n cameo python=2.7 anaconda
conda activate cameo
conda install -c conda-forge opencv=2.4 
conda deactivate
'''
import cv2
import os
path='C:/Users/HP/Desktop/Image'
images_files=os.listdir(path)
#####################################
#                                                                      #
#               Loading and saving an image  ######
#####################################
'''
img = cv2.imread(path+'/'+images_files[0])
cv2.imshow('Input image', img)
#print(type(img))
#<class 'numpy.ndarray'>
cv2.waitKey()
'''
'''
import cv2
gray_img = cv2.imread(path+'/'+images_files[0], cv2.IMREAD_GRAYSCALE)
cv2.imshow('Grayscale', gray_img)
cv2.imwrite(path+'/'+'output.jpg', gray_img)
cv2.waitKey()
'''
#####################################
#                                                                      #
#        Converting between color spaces  ######
#####################################

'''
import cv2
color_spaces_flags= print [x for x in dir(cv2) if x.startswith('COLOR_')]
'''
'''
import cv2
img = cv2.imread(path+'/'+images_files[0])
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('Y channel', yuv_img[:, :, 0])
cv2.imshow('U channel', yuv_img[:, :, 1])
cv2.imshow('V channel', yuv_img[:, :, 2])
cv2.imshow('H channel', hsv_img[:, :, 0])
cv2.imshow('S channel', hsv_img[:, :, 1])
cv2.imshow('V channel', hsv_img[:, :, 2])
cv2.imshow('Grayscale image', gray_img)
cv2.imshow('HSV image', hsv_img)
cv2.waitKey()
'''

#####################################
#                                                                      #
#                                 Image translation ######
#####################################


'''
import cv2
import numpy as np
img = cv2.imread(path+'/'+images_files[0])
num_rows, num_cols = img.shape[:2]
translation_matrix = np.float32([ [1,0,70], [0,1,110] ])
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols,num_rows))
cv2.imshow('Translation', img_translation)
cv2.waitKey()
'''
######################################################
#                                                                                                        #
#        To move the image in the middle of a bigger image frame ######
######################################################


'''
import cv2
import numpy as np
img = cv2.imread(path+'/'+images_files[0])
num_rows, num_cols = img.shape[:2]
translation_matrix = np.float32([ [1,0,70], [0,1,110] ])
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols + 70,num_rows + 110))
translation_matrix = np.float32([ [1,0,-30], [0,1,-50] ])
img_translation = cv2.warpAffine(img_translation, translation_matrix,(num_cols + 70 + 30, num_rows + 110 + 50))
cv2.imshow('Translation', img_translation)
cv2.waitKey()
'''
#####################################
#                                                                      #
#                                   Image Rotation  ######
#####################################


'''
import cv2
import numpy as np
img = cv2.imread(path+'/'+images_files[0])
num_rows, num_cols = img.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
cv2.imshow('Rotation', img_rotation)
cv2.waitKey()
'''
#####################################
#                                                                      #
#        rotation while preventing cropping ######
#####################################


'''
import cv2
import numpy as np
img = cv2.imread(path+'/'+images_files[0])
num_rows, num_cols = img.shape[:2]
translation_matrix = np.float32([ [1,0,int(0.5*num_cols)],[0,1,int(0.5*num_rows)] ])
rotation_matrix = cv2.getRotationMatrix2D((num_cols, num_rows), 30,1)
img_translation = cv2.warpAffine(img, translation_matrix,(2*num_cols, 2*num_rows))
img_rotation = cv2.warpAffine(img_translation, rotation_matrix,(2*num_cols, 2*num_rows))
cv2.imshow('Rotation', img_rotation)
cv2.waitKey()
'''
#####################################
#                                                                      #
#                                     Image Scaling  ######
#####################################


'''
img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation =cv2.INTER_LINEAR)
cv2.imshow('Scaling - Linear Interpolation', img_scaled)
img_scaled =cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Scaling - Cubic Interpolation', img_scaled)
img_scaled =cv2.resize(img,(450, 400), interpolation = cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size', img_scaled)
cv2.waitKey()
'''
#####################################
#                                                                      #
#                         Affine Transformations ######
#####################################


'''
import cv2
import numpy as np
img = cv2.imread(path+'/'+images_files[0])
rows, cols = img.shape[:2]
#To get the mirror image
#src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
#dst_points = np.float32([[cols-1,0], [0,0], [cols-1,rows-1]])
src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
dst_points = np.float32([[0,0], [int(0.6*(cols-1)),0], [int(0.4*(cols-1)),rows-1]])
affine_matrix = cv2.getAffineTransform(src_points, dst_points)
img_output = cv2.warpAffine(img, affine_matrix, (cols,rows))
cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey()
'''
#####################################
#                                                                      #
#                  Projective Transformations  ######
#####################################


'''
import cv2
import numpy as np
img = cv2.imread(path+'/'+images_files[0])
rows, cols = img.shape[:2]
src_points = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
dst_points = np.float32([[0,0], [cols-1,0], [int(0.33*cols),rows-1],[int(0.66*cols),rows-1]])
#src_points = np.float32([[0,0], [0,rows-1], [cols/2,0], [cols/2,rows-1]])
#dst_points = np.float32([[0,100], [0,rows-101], [cols/2,0], [cols/2,rows-1]])

projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
img_output = cv2.warpPerspective(img, projective_matrix, (cols,rows))
cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey()
'''
#####################################
#                                                                      #
#       Image Warping                              ######
#####################################


'''
import cv2
import numpy as np
import math
img = cv2.imread(path+'/'+images_files[0], cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

#####################
# Vertical wave
img_output = np.zeros(img.shape, dtype=img.dtype)
for i in range(rows):
    for j in range(cols):
        offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
        offset_y = 0
        if j+offset_x < rows:
            img_output[i,j] = img[i,(j+offset_x)%cols]
        else:
            img_output[i,j] = 0
cv2.imshow('Input', img)
cv2.imshow('Vertical wave', img_output)
#####################
# Horizontal wave
img_output = np.zeros(img.shape, dtype=img.dtype)
for i in range(rows):
    for j in range(cols):
        offset_x = 0
        offset_y = int(4.0 * math.sin(2 * 3.14 * j / 150))
    if i+offset_y < rows:
        img_output[i,j] = img[(i+offset_y)%rows,j]
    else:
        img_output[i,j] = 0
cv2.imshow('Horizontal wave', img_output)
#####################
# Both horizontal and vertical
img_output = np.zeros(img.shape, dtype=img.dtype)
for i in range(rows):
    for j in range(cols):
        offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
        offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
        if i+offset_y < rows and j+offset_x < cols:
            img_output[i,j] = img[(i+offset_y)%rows,(j+offset_x)%cols]
        else:
            img_output[i,j] = 0
cv2.imshow('Multidirectional wave', img_output)
#####################
# Concave effect
img_output = np.zeros(img.shape, dtype=img.dtype)
for i in range(rows):
    for j in range(cols):
        offset_x = int(128.0 * math.sin(2 * 3.14 * i / (2*cols)))
        offset_y = 0
        if j+offset_x < cols:
            img_output[i,j] = img[i,(j+offset_x)%cols]
        else:
            img_output[i,j] = 0
cv2.imshow('Concave', img_output)
cv2.waitKey()
'''


 
img = cv2.imread('C:/Users/HP/Downloads/opencv-computer_vision/images/flask-crud.jpg', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ',img.shape)
 
#scale_percent = 20 # percent of original size
#width = int(img.shape[1] * scale_percent / 100)
#height = int(img.shape[0] * scale_percent / 100)
width=480
height=300
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.imwrite('C:/Users/HP/Downloads/opencv-computer_vision/images/flask-crud-resized.jpg', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
















