import cv2
import numpy as np
img_folder_path='C:/Users/HP/Downloads/opencv-computer_vision/images'
#Detecting the corners
'''
img = cv2.imread(img_folder_path+'/box.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 4,5, 0.04) # to detect only sharp corners
#dst = cv2.cornerHarris(gray, 14, 5, 0.04) # to detect soft corners
# Result is dilated for marking the corners
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01*dst.max()] = [0,0,0]
cv2.imshow('Harris Corners',img)
cv2.waitKey()
'''
#Good Features to track
'''

img = cv2.imread(img_folder_path+'/box.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 7, 0.05, 25)
corners = np.float32(corners)
for item in corners:
    x, y = item[0]
cv2.circle(img, (x,y), 5, 255, -1)
cv2.imshow("Top 'k' features", img)
cv2.waitKey()
'''
#Scale Invariant Feature Transform (SIFT)
'''
input_image = cv2.imread(img_folder_path+'/box.jpg')
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT()
keypoints = sift.detect(gray_image, None)
input_image = cv2.drawKeypoints(input_image, keypoints,outImage=None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT features', input_image)
cv2.waitKey()
'''
#Speeded Up Robust Features (SURF)
'''
img = cv2.imread(img_folder_path+'/box.jpg')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
surf = cv2.SURF()
# This threshold controls the number of keypoints
surf.hessianThreshold = 15000
kp, des = surf.detectAndCompute(gray, None)
img = cv2.drawKeypoints(img, kp, None, (0,255,0), 4)
cv2.imshow('SURF features', img)
cv2.waitKey()
'''
#Features from Accelerated Segment Test(FAST)
'''
gray_image = cv2.imread(img_folder_path+'/wrench.png', 0)
cv2.imshow("Gray image",gray_image)

fast = cv2.FastFeatureDetector()
# Detect keypoints
keypoints = fast.detect(gray_image, None)
print("Number of keypoints with non max suppression:", len(keypoints))
# Draw keypoints on top of the input image
img_keypoints_with_nonmax = cv2.drawKeypoints(gray_image, keypoints, color=(0,255,0))
cv2.imshow('FAST keypoints - with non max suppression',img_keypoints_with_nonmax)

# Disable nonmaxSuppression
fast.setBool('nonmaxSuppression', False)
# Detect keypoints again
keypoints = fast.detect(gray_image, None)
print("Total Keypoints without non max suppression:", len(keypoints))
# Draw keypoints on top of the input image
img_keypoints_without_nonmax = cv2.drawKeypoints(gray_image, keypoints,color=(0,255,0))
cv2.imshow('FAST keypoints - without non max suppression',img_keypoints_without_nonmax)
cv2.waitKey()
cv2.destroyAllWindows()
'''

#Binary Robust Independent Elementary Features (BRIEF)
'''
gray_image = cv2.imread(img_folder_path+'/wrench.png', 0)
# Initiate FAST detector
fast = cv2.FastFeatureDetector()
# Initiate BRIEF extractor
brief = cv2.DescriptorExtractor_create("BRIEF")
# find the keypoints with STAR
keypoints = fast.detect(gray_image, None)
# compute the descriptors with BRIEF
keypoints, descriptors = brief.compute(gray_image, keypoints)
gray_keypoints = cv2.drawKeypoints(gray_image, keypoints, None,color=(0,255,0))
cv2.imshow('BRIEF keypoints', gray_keypoints)
cv2.waitKey()
'''
#Oriented FAST and Rotated BRIEF (ORB)
'''
input_image =  cv2.imread(img_folder_path+'/input.jpg')
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
# Initiate ORB object
orb = cv2.ORB()
# find the keypoints with ORB
keypoints = orb.detect(gray_image, None)
# compute the descriptors with ORB
keypoints, descriptors = orb.compute(gray_image, keypoints)
# draw only the location of the keypoints without size or orientation
final_keypoints = cv2.drawKeypoints(input_image, keypoints,None, color=(0,255,0), flags=0)
cv2.imshow('ORB keypoints', final_keypoints)
cv2.waitKey()
'''



















