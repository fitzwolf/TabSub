import numpy as np
import cv2
from tensorflow import keras
import os

def predict_character_best(image_path, model):
    '''
        predict_character_best: This attempts to predict the most likely match for the image provided
        image_path: path to the image of interest
        model: pretrained model

        return:
        predicted_character: best match based on model and image provided
        confidence: confidence score for the predicted character
    '''
    # Load and preprocess the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (28, 28))
    image_resized = 255 - image_resized  # Invert the image (EMNIST has white characters on a black background)
    image_resized = image_resized.astype("float32") / 255.0
    image_input = np.expand_dims(image_resized, axis=(0, -1))
   
    # Make a prediction using the model
    prediction = model.predict(image_input)

    # Post-process the prediction to get the recognized character and its confidence
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Create a mapping of class indices to characters
    class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    # Get the predicted character from the class mapping
    print(predicted_class)
    predicted_character = class_mapping[predicted_class]

    return predicted_character, confidence



def predict_character(image_path, model):
    '''
        predict_character: This attempts to predict the three most likely matches for the image provided
        image_path: path to the image of interest
        model: pretrained model

        return:
        top3_character: best three matches based on model and image provided
        top3_confidence: confidence score for the predicted characters
    '''
    # Load and preprocess the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (28, 28))
    image_resized = 255 - image_resized  # Invert the image (EMNIST has white characters on a black background)
    image_resized = image_resized.astype("float32") / 255.0
    image_input = np.expand_dims(image_resized, axis=(0, -1))

    # Make a prediction using the model
    prediction = model.predict(image_input)

    # Post-process the prediction to get the recognized character and its confidence
    top3_indices = np.argsort(prediction[0])[-3:][::-1]
    top3_confidences = prediction[0][top3_indices]

    # Create a mapping of class indices to characters
    # class_mapping = '0123456789abcdefghijklmnopqrstuvwxyz'
    class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    # Get the predicted character from the class mapping
    top3_characters = [class_mapping[i] for i in top3_indices]
    
    return top3_characters, top3_confidences

# Load the saved model
modeldir = "models_classifier"
modelfn = "emnist_letter_detection_optimized.h5"
model = keras.models.load_model(os.path.join(modeldir, modelfn))

# Load an input image
imgdir = "cropped_images"
imglist = ["image0.png", "image1.png", "image2.png", "image3.png"]

# Predict the character and its confidence
for image in imglist:
    img = os.path.join(imgdir, image)
    top3_predicted_characters, top3_confidences = predict_character(img, model)
    print(f"Top 3 Predicted characters: {top3_predicted_characters}, Actual: {image}")
    print(f"Top 3 Confidences: {top3_confidences}")


#### this will use only the best predicted character
# # Predict the character and its confidence
# for image in imglist:
#     img = os.path.join(imgdir, image)
#     predicted_character, confidence = predict_character(img, model)
#     print(f"Predicted character: {predicted_character}, Actual: {image}")
#     print(f"Confidence: {confidence:.2f}")