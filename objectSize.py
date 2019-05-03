
#for small cars max width and length is 4.1 meters (161 inches) and 1.8 Meters(71 inches) 
#	examples:Volkswagen Polo,Ford Fiesta,Hundai i20,Fiat Punto etc

#for city cars() max width and length is 4 meters (144 inches) and 1.8 Meters(71 inches)
# 	examples:

#for sedan cars(large) max width and length is 6 meters (236 inches) and 2.2 Meters(87 inches)
from matplotlib import pyplot as plt
import math
# import the necessary packages
from scipy.spatial import distance as dist
#imutis is for basic image processing functions such as translation, rotation, resizing, skeletonization, etc
import imutils
from imutils import perspective
from imutils import contours

#numpy package is used for scientific computing
import numpy as np
#argparse is used for user-friendly command-line interface.
import argparse
#importing open computer vision package.
import cv2
number=0
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in meters)")
ap.add_argument("-l", "--totalSize", required=True,
	help="total area in meter Squares")
args = vars(ap.parse_args())

#cascade_src = 'cars.xml'
#video_src = 'dataset/parking5.JPG'
#video_src = 'dataset/video2.avi'

#cap = cv2.imread(args["image"])
#car_cascade = cv2.CascadeClassifier(cascade_src)

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.GaussianBlur(gray, (5, 9), 0)
#80 95
gray = cv2.bilateralFilter(gray,9,85,95)
#SVM & Principal Component analysis
# sigmaColor – Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.
# sigmaSpace – Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace
#gray = cv2.medianBlur(gray,7, 7)

# perform edge detection, then perform a dilation +// erosion to
# close gaps in between object edges
#60 1804
edged = cv2.Canny(gray, 55, 185)


edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.dilate(edged, None, iterations=1)

#edged = cv2.erode(edged, None, iterations=1)
#edged = cv2.morphologyEx(edged, cv2.MORPH_GRADIENT, kernel)

#cars = car_cascade.detectMultiScale(edged, 1.1, 1)

    #for (x,y,w,h) in cars:
    #    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)      


plt.subplot(121),plt.imshow(gray,cmap = 'gray')
plt.title('GrayScale Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edged,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

#cv2.imshow("Image", edged)
# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None
idx = 0
TotalCarArea=0
length=0
width=0
avgCars=0
# loop over the contours individually
for c in cnts:
	
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 1400:
		continue
	x,y,w,h = cv2.boundingRect(c)
	if w>50 and h>50:
		idx+=1
		new_img=image[y:y+h,x:x+w]
		cv2.imwrite(str(idx) + '.png', new_img)
	# if cv2.contourArea(c) > 000:
	# 	continue
	
	
	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	# loop over the original points and draw them
	for (x, y) in box:
		number= number+1
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1) 


	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]

	# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric
	#TotallengthOfCars = TotallengthOfCars + dimA
	#TotalWidthOfCars = TotalWidthOfCars + dimB
	#print("Length x",dimA,"Length Y",dimB)
	TotalCarArea = TotalCarArea + int(dimA*dimB)

	# draw the object sizes on the image
	cv2.putText(orig, "{:.1f}".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}ms".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

	# show the output image
	
	cv2.imshow("Image ", orig)
	cv2.waitKey(0)
	a = number/4
print("Total Cars:",a)

#200msq is approx dimension

#print("Length & width of total area is ",length,width," Respectively")

totalSize = int(args["totalSize"])
print("Total Area: ",totalSize, "square meters")

#TotalCarArea = TotallengthOfCars*TotalWidthOfCars
print("Approximate Area covered By Cars: ",TotalCarArea, "square meters")

totalSize = math.floor(totalSize-TotalCarArea)
print("Approximate Remaing Area: ",totalSize, "square meters")
#15.6 is the approximate average area of car
avgCars = (totalSizmde/15.6)
avgCars = math.floor(avgCars)
print("Approximate Number of Cars than can be accodomated:",avgCars)

#to run the code  " python objectSize.py --image parking5.jpg --width 2.3  --totalSize 200 "
																					#mention width in meters 

