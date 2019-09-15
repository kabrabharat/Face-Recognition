from imutils.video import VideoStream
from imutils.video import FPS
from imutils import face_utils, resize
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import time


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-c", "--cascade", required=True, help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

print(" --- loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
#detector = cv2.CascadeClassifier(args["cascade"])
detector = face_recognition.api.dlib.get_frontal_face_detector()
# initialize the video stream and allow the camera sensor to warm up
print(" --- starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
 
# start the FPS counter
fps = FPS().start()

count = 0
unknown_count = {}
flag = 0
flag1 = 0
detecthigh = 0
while True:
	
	boxes = []

	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	frame = resize(frame, width=750)
	
	# update frame
	fps.update()
	
	# convert the input frame from (1) BGR to grayscale (for face
	# detection) and (2) from BGR to RGB (for face recognition)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
	# detect faces in the grayscale frame
	#rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
	rects = detector(gray, 1)
	for i, rect in enumerate(rects):
		(x,y,w,h) = face_utils.rect_to_bb(rect)
		#cv2.rectangle(rgb,(x,y),(x+w,y+h),(0,255,0),2)   
	# OpenCV returns bounding box coordinates in (x, y, w, h) order
	# but we need them in (top, right, bottom, left) order, so we
	# need to do a bit of reordering
		box = [y, x + w, y + h, x]
		boxes.append(box)
	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(rgb, boxes, num_jitters = 3)
	names = []

	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance = 0.50)
		name = "Unknown"

		if name not in unknown_count.keys():
			unknown_count[name] = 0
		else:
			unknown_count[name] += 1 

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
 
			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			
				# determine the recognized face with the largest number
				# of votes (note: in the event of an unlikely tie Python
				# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
			#name = [n for n,m in counts.items() if m>=55]
#			if counts[name]>=60:
#
#				name = [n for n,m in counts.items() if m>=20]
#				name = " ".join(name).split()[0]
#				print(name,counts[name])
#			else:
#				continue
			flag += 1

			if flag>=1:
				print(name," Detected", time.ctime(time.time()))
				flag = 0

			else:
				name = "Unknown"
				print(name)
				detecthigh = 0
				


		else:
			flag1 += 1
		
		if flag1>=1:
			print(name, " Detected", time.ctime(time.time()))
			detecthigh = 0
			flag1 = 0
			unknown_count = {}
			


		# update the list of names
		names.append(name)
	for ((y, a, b, x), name) in zip(boxes, names):
		# draw the predicted face name on the image
		#cv2.rectangle(frame, (left, top), (right, bottom),
		#	(0, 255, 0), 2)
		#y = top - 15 if top - 15 > 15 else top + 15
		cv2.rectangle(frame,(x,y),(a,b),(255,0,0),2)   
		cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 2)
 

	detecthigh += 1


	

	# display the image to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
	if key == ord("s"):
		unknown_count = {}
		flag = 0
		flag1 = 0


