import cv2
import numpy as np


# function to display the coordinates of
# of the points clicked on the image 
count = 2
arr = []
top_left = (0,0)
bottom_right=(0,0)
i = 0 


#source: https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
#Prompts a window and asked the user to click top left and bottom right corner of their first letter.
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
    
#Process the points to see which one is bottom right or top left
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

#crop the image and output it as a png in the cropped_images folder
def crop_image(tl,br, pad):
    global img_copy, i 
    h,w,c = img_copy.shape

    #only y needs padding because width is most likely gonna be the same -> w is the longest, thus add 3 pixel.
    if(tl[1]-pad <= 0):
        pad_y = 0
    else:
        pad_y = tl[1]-pad
    if(br[0]+7 >= w):
        pad_x = w
        print("im here")
    else:
        pad_x = br[0]+7
    
    crop_img = img_copy[pad_y:br[1]+7, tl[0]:pad_x].copy()
    cv2.imwrite(f'./cropped_images/image{i}.png', crop_img)
    i+=1

#update the coordinate to get the next letter
def update_coor(tl,br):
    global img_copy
    h,w,c = img_copy.shape

    new_tl = (br[0], tl[1])
    new_br = ( (2*br[0] - tl[0]) ,  br[1])

    if(new_br[0] >= w):
        new_br = (w,br[1])
    if(new_br[0] - new_tl[0] < (br[0]-tl[0])):
        return (-1,-1)
    return new_tl,new_br

#main
if __name__=="__main__":
    img = cv2.imread('world.jpeg', 1)
    img_copy = img.copy()
    print("Please click on the top-left corner and the bottom-right corner of your first letter.")
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    process_points(arr)
    loop=6
    i=0
    while(True):
        #crop the image and output to cropped_images
        crop_image(top_left,bottom_right,35) 
        
        # TODO:
        # Change if statement to be: 
        # if image (i-1) and block = background:
            #break
        if loop==0:
            break
        else:
            top_left,bottom_right = update_coor(top_left,bottom_right)
            #We are at the end of the picture
            if(top_left == -1):
                print("Nothing left to read")
                break
            loop-=1

        
        

