import cv2
import numpy as np
#Detecting and tracking faces
folder_path='C:/Users/HP/Anaconda3/envs/cameo/Library/etc/haarcascades'
face_path=folder_path+'/haarcascade_frontalface_alt.xml'
eye_path=folder_path+'/haarcascade_eye.xml'
img_folder_path='C:/Users/HP/Downloads/opencv-computer_vision/images'
'''
face_cascade =cv2.CascadeClassifier(face_path)
cap = cv2.VideoCapture(0)
scaling_factor = 0.5
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    cv2.imshow('Face Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()
'''
#Fun with faces
'''
face_cascade =cv2.CascadeClassifier(face_path)
face_mask = cv2.imread('mask_hannibal.png')
h_mask, w_mask = face_mask.shape[:2]
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
cap = cv2.VideoCapture(0)
scaling_factor = 0.5
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in face_rects:
        if h > 0 and w > 0:
        # Adjust the height and weight parameters depending on the sizes and the locations. You need to play around with these to make sure you get it right.
            h, w = int(1.4*h), int(1.0*w)
            y -= 0.1*h
            # Extract the region of interest from the image
            frame_roi = frame[y:y+h, x:x+w]
            face_mask_small = cv2.resize(face_mask, (w, h),interpolation=cv2.INTER_AREA)
            # Convert color image to grayscale and threshold it
            gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_mask, 180, 255,cv2.THRESH_BINARY_INV)
            # Create an inverse mask
            mask_inv = cv2.bitwise_not(mask)
            # Use the mask to extract the face mask region of interest
            masked_face = cv2.bitwise_and(face_mask_small, face_mask_small,mask=mask)
            # Use the inverse mask to get the remaining part of the image
            masked_frame = cv2.bitwise_and(frame_roi, frame_roi,mask=mask_inv)
            # add the two images to get the final output
            frame[y:y+h, x:x+w] = cv2.add(masked_face, masked_frame)
    cv2.imshow('Face Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:break
cap.release()
cv2.destroyAllWindows()
'''
#Detecting eyes
'''
face_cascade =cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')
cap = cv2.VideoCapture(0)
ds_factor = 0.5
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (x_eye,y_eye,w_eye,h_eye) in eyes:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            color = (0, 255, 0)
            thickness = 3
            cv2.circle(roi_color, center, radius, color, thickness)
    cv2.imshow('Eye Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()
'''
#Fun with eyes
'''
face_cascade =cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')
img_folder_path='C:/Users/HP/Downloads/opencv-computer_vision/images'
img = cv2.imread(img_folder_path+'/input.jpg')
sunglasses_img = cv2.imread(img_folder_path+'/sunglasses.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
centers = []
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (x_eye,y_eye,w_eye,h_eye) in eyes:
        centers.append((x + int(x_eye + 0.5*w_eye), y + int(y_eye +0.5*h_eye)))
if len(centers) > 0:
# Overlay sunglasses; the factor 2.12 is customizable depending on the size of the face
    sunglasses_width = 2.12 * abs(centers[1][0] - centers[0][0])
    overlay_img = np.ones(img.shape, np.uint8) * 255
    h, w = sunglasses_img.shape[:2]
    scaling_factor = sunglasses_width / w
    overlay_sunglasses = cv2.resize(sunglasses_img, None,fx=scaling_factor,fy=scaling_factor, interpolation=cv2.INTER_AREA)
    x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]
    # customizable X and Y locations; depends on the size of the face
    x -= 0.26*overlay_sunglasses.shape[1]
    y += 0.85*overlay_sunglasses.shape[0]
    h, w = overlay_sunglasses.shape[:2]
    overlay_img[y:y+h, x:x+w] = overlay_sunglasses
    # Create mask
    gray_sunglasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray_sunglasses, 110, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    temp = cv2.bitwise_and(img, img, mask=mask)
    temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
    final_img = cv2.add(temp, temp2)
    cv2.imshow('Eye Detector', img)
    cv2.imshow('Sunglasses', final_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
'''
#Detecting ears
'''
left_ear_cascade = cv2.CascadeClassifier(folder_path+'/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier(folder_path+'/haarcascade_mcs_rightear.xml')
if left_ear_cascade.empty():
    raise IOError('Unable to load the left ear cascade classifier xml file')
if right_ear_cascade.empty():
    raise IOError('Unable to load the right ear cascade classifier xml file')

img = cv2.imread(img_folder_path+'/input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
left_ear = left_ear_cascade.detectMultiScale(gray, 1.3, 5)
print(left_ear)
right_ear = right_ear_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in left_ear:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
for (x,y,w,h) in right_ear:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)
cv2.imshow('Ear Detector', img)
cv2.waitKey()
cv2.destroyAllWindows()
'''
#Detecting a mouth
'''
mouth_cascade =cv2.CascadeClassifier(folder_path+'/haarcascade_mcs_mouth.xml')
if mouth_cascade.empty():
    raise IOError('Unable to load the mouth cascade classifier xml file')
cap = cv2.VideoCapture(0)
ds_factor = 0.5
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    for (x,y,w,h) in mouth_rects:
        y = int(y - 0.15*h)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        break
    cv2.imshow('Mouth Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:break
cap.release()
cv2.destroyAllWindows()
'''
#It’s time for a moustache
'''
mouth_cascade = cv2.CascadeClassifier(folder_path+'/haarcascade_mcs_mouth.xml')
moustache_mask = cv2.imread(img_folder_path+'/moustache.png')
h_mask, w_mask = moustache_mask.shape[:2]
if mouth_cascade.empty():raise IOError('Unable to load the mouth cascade classifier xml file')
cap = cv2.VideoCapture(0)
scaling_factor = 0.5
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.3, 5)
    if len(mouth_rects) > 0:
        (x,y,w,h) = mouth_rects[0]
        h, w = int(0.6*h), int(1.2*w)
        x -= 0.05*w
        y -= 0.55*h
        frame_roi = frame[y:y+h, x:x+w]
        moustache_mask_small = cv2.resize(moustache_mask, (w, h),
        interpolation=cv2.INTER_AREA)
        gray_mask = cv2.cvtColor(moustache_mask_small, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_mask, 50, 255,cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        masked_mouth = cv2.bitwise_and(moustache_mask_small,moustache_mask_small, mask=mask)
        masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
        frame[y:y+h, x:x+w] = cv2.add(masked_mouth, masked_frame)
    cv2.imshow('Moustache', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()
'''
#Detecting a nose
'''
nose_cascade =cv2.CascadeClassifier(folder_path+'/haarcascade_mcs_nose.xml')
if nose_cascade.empty():raise IOError('Unable to load the nose cascade classifier xml file')
cap = cv2.VideoCapture(0)
ds_factor = 0.5
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in nose_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        break
    cv2.imshow('Nose Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:break
cap.release()
cv2.destroyAllWindows()
'''
#Detecting pupils
import math
img = cv2.imread(img_folder_path+'/input.jpg')
scaling_factor = 0.7
img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor,interpolation=cv2.INTER_AREA)
cv2.imshow('Input', img)
gray = cv2.cvtColor(~img, cv2.COLOR_BGR2GRAY)
ret, thresh_gray = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
for contour in contours:
    area = cv2.contourArea(contour)
    rect = cv2.boundingRect(contour)
    x, y, width, height = rect
    radius = 0.25 * (width + height)
    area_condition = (100 <= area <= 200)
    symmetry_condition = (abs(1 - float(width)/float(height)) <= 0.2)
    fill_condition = (abs(1 - (area / (math.pi * math.pow(radius, 2.0))))<= 0.3)
    if area_condition and symmetry_condition and fill_condition:
        cv2.circle(img, (int(x + radius), int(y + radius)),int(1.3*radius), (0,180,0), -1)
cv2.imshow('Pupil Detector', img)
c = cv2.waitKey()
cv2.destroyAllWindows()








