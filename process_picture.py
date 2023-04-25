import cv2
import numpy as np


# function to display the coordinates of
# of the points clicked on the image 
count = 2
arr = []
top_left =(0,0)
bottom_right=(0,0)
i = 0 


#source: https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
def click_event(event, x, y, flags, params):
    
    # checking for left mouse clicks
    global count
    global img
    global arr

    if event == cv2.EVENT_LBUTTONDOWN:
        if count == 0:
            print("Thank you for inputing the 2 inputs. Please press any key to close the window.")
            return
        print(x, ' ', y)
        arr.append([x,y])
        count-=1
        img = cv2.circle(img, (x,y), radius=2, color=(0, 0, 255), thickness=-1)
        cv2.imshow('image', img)
  
    # checking for right mouse clicks     
    if event==cv2.EVENT_RBUTTONDOWN:
        if count == 0:
            print("Thank you for inputing the 2 inputs. Please press any key to close the window.")
            return
        print(x, ' ', y)
        arr.append([x,y])
        count-=1
        img = cv2.circle(img, (x,y), radius=2, color=(0, 0, 255), thickness=-1)
        cv2.imshow('image', img)
    
    
def process_points(arr):
    global bottom_right
    global top_left
    temp1 = arr[0]
    temp2 = arr[1]   
    if(temp1[0] < temp2[0]):
        top_left = temp1
        bottom_right = temp2
    else:
        top_left = temp2
        bottom_right = temp1

def crop_image(tl,br, i):
    global img_copy
    crop_img = img_copy[tl[1]:br[1], tl[0]:br[0]].copy()
    cv2.imwrite(f'./cropped_images/image{i}.png', crop_img)
    i+=1

if __name__=="__main__":
    img = cv2.imread('handwriting.png', 1)
    img_copy = img.copy()
    print("Please click on the top-left corner and the bottom-right corner of your first letter.")
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    process_points(arr)
    crop_image(top_left,bottom_right,i)

