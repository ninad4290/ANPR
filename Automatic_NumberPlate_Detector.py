#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr


# In[30]:


img = cv2.imread('image1.jpg')


# In[31]:


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray,cv2.COLOR_BGR2RGB))


# In[32]:


bfilter = cv2.bilateralFilter(gray,11,17,17) #filtering 
edged = cv2.Canny(bfilter,30,200) #edge detection
plt.imshow(cv2.cvtColor(edged,cv2.COLOR_BGR2RGB))


# In[33]:


keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#find contours approximates to 4 keypoints  to rectangles
contours = imutils.grab_contours(keypoints) #grabs contours
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] #top 10 contours returned


# In[34]:


location = None                       #loop through each of them to check if its a square (4keypoints) or cicrle
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True) #higher the number the more rough the estimation will be
    if len(approx) == 4:
        location = approx
        break


# In[35]:


location


# In[36]:


mask = np.zeros(gray.shape, np.uint8) #blank mask
new_image = cv2.drawContours(mask, [location], 0,255, -1)#used the contour on the blank mask
new_image = cv2.bitwise_and(img, img, mask=mask)


# In[37]:


plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))


# In[38]:


(x,y) = np.where(mask==255) #finding places where image isnt black
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]


# In[39]:


plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))


# In[40]:


reader = easyocr.Reader(['en']) # read text using easy ocr
result = reader.readtext(cropped_image)
result


# In[41]:


text = result[0][-2]  #rendering result on the image (2nd last string of the array) 
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))


# In[ ]:




