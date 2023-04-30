import cv2
import numpy as np

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_image_blocks(block1, block2, threshold=1000):
    # Check if both blocks have the same shape and are 96x96
    # if block1.shape != (96, 96) or block2.shape != (96, 96):
    #     raise ValueError("Both image blocks must have the shape (96, 96)")

    if block1.shape != block2.shape:
        raise ValueError("Both image blocks must have the shape")
    # Convert the image blocks to grayscale
    gray_block1 = cv2.cvtColor(block1, cv2.COLOR_BGR2GRAY)
    gray_block2 = cv2.cvtColor(block2, cv2.COLOR_BGR2GRAY)

    # Calculate the mean squared error between the two grayscale image blocks
    mse_value = mse(gray_block1, gray_block2)

    # Compare the mean squared error with the threshold value
    if mse_value <= threshold:
        return "similar or the same"
    else:
        return "dissimilar"
