# Final Project

## To setup dependencies
```
pip install -r requirements.txt
```

## Using Camera.py (A Webcam in required)
To use Camera.py
```
python Camera.py
```


## Running process_picture.py

1. Run the Python File
2. A window will pop out, click on top-left and bottom-right conner of your first letter.
3. Press any key to exit the window
4. Cropped images will be under "cropped_images" folder.

## Running predict_character.py

You may need to edit this file to change the path to the images you wish to edit, in a later version this should
become a passed in value but for now hard edit these 2 lines:

imgdir = "cropped_images"
imglist = ["image0.png", "image1.png", "image2.png", "image3.png"]

Once you have those set to the images you want to predict on then

python predict_character.py

This will return the 3 most likely characters and their confidence

example output:
1/1 [==============================] - 0s 119ms/step

Top 3 Predicted characters: ['c', 'C', 'N'], Actual: image0.png

Top 3 Confidences: [0.1447299  0.13384989 0.09934802]

1/1 [==============================] - 0s 20ms/step

Top 3 Predicted characters: ['d', 'S', 'a'], Actual: image1.png

Top 3 Confidences: [0.69721633 0.05965977 0.0371861 ]

1/1 [==============================] - 0s 19ms/step

Top 3 Predicted characters: ['m', 'M', 'A'], Actual: image2.png

Top 3 Confidences: [0.32302594 0.25186086 0.18663743]

1/1 [==============================] - 0s 21ms/step

Top 3 Predicted characters: ['j', 'i', '1'], Actual: image3.png

Top 3 Confidences: [0.4460022  0.40011805 0.03216499]
