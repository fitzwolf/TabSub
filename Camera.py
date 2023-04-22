# import the opencv library
# some starter code to read from webcam
# source: https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/


import cv2
import mediapipe as mp

# MediaPipe Hand Detection ReadMe: https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

def draw_index_finger(image, hand_landmarker_result):
	"""Draws the index finger tip location."""
	if not hand_landmarker_result or not hand_landmarker_result.multi_hand_landmarks:
		return image
	image_rows, image_cols, _ = image.shape
	for hand_landmarks in hand_landmarker_result.multi_hand_landmarks:
		index_finger_tip = mp_drawing._normalized_to_pixel_coordinates(
			hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
			hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y, image_cols,
			image_rows)
		cv2.circle(image, index_finger_tip, 5, (255, 0, 0), -1)
	return image

def draw_hand_landmarks(image, hand_landmarker_result):
	if not hand_landmarker_result or not hand_landmarker_result.multi_hand_landmarks:
		return image
	for hand_landmarks in hand_landmarker_result.multi_hand_landmarks:
		mp_drawing.draw_landmarks(
			image,
			hand_landmarks,
			mp_holistic.HAND_CONNECTIONS,
			landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
			connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
	return image


# define a video capture object
# 0 is the device #, it can be confusing but if you have multiple potential sources
# you might have to mess with this value.
vid = cv2.VideoCapture(0)

#with HandLandmarker.create_from_options(options) as landmarker:
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
	while(True):	
		# Capture the video frame
		# by frame
		ret, frame = vid.read()
		fps = float(vid.get(cv2.CAP_PROP_FPS))

		if not ret:
			print("Failed to grab frame")
			continue

		# Display the resulting frame
		# could instead save these frames to make a non-live video
		frame = cv2.flip(frame, 1)
		cv2.imshow('frame', frame)
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = hands.process(image)
		cv2.imshow('hand', cv2.cvtColor(draw_hand_landmarks(image.copy(), results), cv2.COLOR_RGB2BGR))
		cv2.imshow('index finger', cv2.cvtColor(draw_index_finger(image.copy(), results), cv2.COLOR_RGB2BGR))
		
		
		# the 'q' button is set as the
		# quitting button you may use any
		# desired button of your choice
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
