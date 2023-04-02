import cv2

import main


def test_main():
    test_image = cv2.imread("data/images/test_image_w_pen_w_noise.png")
    hsv_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
    thresholded_paper = cv2.inRange(hsv_image, (43, 38, 141), (99, 255, 255))
    contours = main.get_contours(thresholded_paper)
    print(len(contours))
    contours = main.process_contours(contours, max_merge_distance=10)
    print(len(contours))
    cv2.imshow("Name", cv2.drawContours(test_image, contours, -1, (0, 255, 0), 3))
    cv2.waitKey(0)

if __name__ == "__main__":
    test_main()
