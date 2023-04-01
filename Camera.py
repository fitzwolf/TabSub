# import the opencv library
# some starter code to read from webcam
# source: https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/


import cv2


# define a video capture object
# 0 is the device #, it can be confusing but if you have multiple potential sources
# you might have to mess with this value.
vid = cv2.VideoCapture(0)

while(True):
	
	# Capture the video frame
	# by frame
	ret, frame = vid.read()

	# Display the resulting frame
	# could instead save these frames to make a non-live video
	cv2.imshow('frame', frame)
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
