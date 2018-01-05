import cv2
import numpy as np

waldo = cv2.imread("waldo.jpg")
world = cv2.imread("wheresWaldo.jpg")
# get width and height
# could also do w,h = waldo.shape[:-1]
w,h = waldo.shape[:2]

# Template matching is scale invariant so it matches a templates exact
# scale to an images exact scale, if either is off, they won't match
# cv2.matchTemplate returns a correlation map, essentially a grayscale image, where each
# pixel denotes how much does the neighbourhood of that pixel match with template
result = cv2.matchTemplate(world,waldo,cv2.TM_CCOEFF)
# minMaxLoc returns the maxmin intensity values in the correlation map (max is highest likely matching place)
# minMaxLoc also returns max/min location of the highest/lowest intensity corrdinates
(minVal,maxVal,minLoc,maxLoc) = cv2.minMaxLoc(result)

# assume maxLoc is the top left coordinate of the matching area
topLeft = maxLoc
botRight = (topLeft[0] + h, topLeft[1] + w)
cv2.rectangle(world,topLeft, botRight, 255, 2)
cv2.imshow("Where's Waldo?", world)
cv2.waitKey(0)

# if we want to match a template to an image that isn't the exact size, we can
# incrementally change the size of the image to match the template
# Sample code:
# loop over the scales of the image, scale from 0.2 to 1.0 with 20 samples
#gray = cv2.cvtColor(world,cv2.COLOR_BGR2GRAY)
#for scale in np.linspace(0.2, 1.0, 20)[::-1]:
#    # resize the image according to the scale, and keep track
#	# of the ratio of the resizing
#    resized = cv2.resize(gray, (gray.shape[1] * scale), interpolation=cv2.INTER_CUBIC)
#    r = gray.shape[1] / float(resized.shape[1])