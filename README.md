# Webcam-Stylus

## To setup dependencies
```
pip install -r requirements.txt
```

## Using hsv.py to find the correct hsv range for control points.
```
python hsv.py --path="path to image" # Opens an image view with hsv sliders
python hsv.py --camera="web camera id" # Opens a webcam view with hsv sliders
```

## Using main.py (A Webcam in required)
To use main.py you need a HSV range for the paper control points and the pen control point. These can be found using hsv.py. The HSV ranges should be placed in 
the main function where cv2.Range is called for the paper and pen.
```
python main.py # To run the program
```