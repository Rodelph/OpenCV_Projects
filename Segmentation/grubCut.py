'''
Calculating a disparity map is a useful wat to segment the foreground and background of an image, but StereoSGBM is not the only algorithm that can accomplish
this and in fact this algorithm is more about gatheriong three dimensional information from two dimensional pictures than anything else. GrabCut however, is a perfect
tool for foreground/ background segementation. The GrabCut algorithm consists of the following steps.
 - A rectangle including the subject(s) of the picture is defined
 - The area lying outisde the rectangle is automatically defined as a background
 - The data contained in the background is used as a reference to distinguish background areas from foreground areas within the user defined rectangle
 - A Gaussion Mixture Model (GMM) models the foreground and background, and labels undefined pixels as probable background and probable foreground.
 - Each pixel in the image is vitually connected to the surrounding pixels through virtual edges, and each edge is assigned a probablility of being foreground 
   or backgoround, based on how similar it is in color to the pixels surroinding it.
 - Each pixel is connected to either a foreground or background node.
 - After the nodes have been connected to either terminal (the background or foreground, also called the source or sink, respectively), the edges between nodes 
   belonging to different terminals are cut (hence the name, GrabCut). Thus, the image is segmented into two parts.  
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

original = cv2.imread("./res/1.jpg")
img = original.copy()
mask = np.zeros(img.shape[:2], np.uint8)

#Creation of zero filled background and foreground models

bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

#We initialize the GrabCut algorithm with arectangle identifying the subject we want to isolate. Thus background and forground models are going to be determined 
# based on the areas left out of the initial rectangle. This rectangle is defined in the next line. 

rect = (50, 170, 421, 378)

#We run the grabcut algorithm? As arguments, we specify the emoty models, the mask, and the rectangle that we want to use to initialize the operation.
#The 5 integer argument is the number of iterrations the algorithm is going to run on the image. You can increase it but at some point pixel classificatiosn will
#converge so, effectively, you might just be adding iterations without any further improvements to the result.

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 50 , cv2.GC_INIT_WITH_RECT)

#To visualize the result of the GrabCut, we want to paint the background black and leave the foreground unchanged. We can make another mask to help us do this. 
#The values :
# 0 => Obvious background pixel / 1 => Obvious forground pixel / 2 => Probable background pixel / 3 => Probable foreground pixel 


mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
img = img*mask2[:,:,np.newaxis]

#Script to display the images 
plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("grabCut")
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("original")
plt.xticks([])
plt.yticks([])

plt.show()