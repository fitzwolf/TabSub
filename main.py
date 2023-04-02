import numpy as np
import cv2
import os
import time

# type aliases
Points = np.ndarray[np.ndarray[int]]
Matrix = np.ndarray[np.ndarray[np.generic]]

def get_contours(img: cv2.Mat) -> tuple[cv2.Mat]:
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_contour_area(contour: cv2.Mat):
    return cv2.contourArea(contour)

# find the distance between two contours
def get_contour_distance(contour1: cv2.Mat, contour2: cv2.Mat) -> float:
    M1 = cv2.moments(contour1)
    M2 = cv2.moments(contour2)
    cx1 = int(M1['m10']/(M1['m00'] + 0.001))
    cy1 = int(M1['m01']/(M1['m00'] + 0.001))
    cx2 = int(M2['m10']/(M2['m00'] + 0.001))
    cy2 = int(M2['m01']/(M2['m00'] + 0.001))
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

def get_contour_distance_matrix(contours: tuple[cv2.Mat]) -> Matrix:
    num_contours = len(contours)
    matrix = np.zeros((num_contours, num_contours))
    for i in range(num_contours):
        for j in range(num_contours):
            matrix[i, j] = get_contour_distance(contours[i], contours[j])
            if i == j:
                matrix[i, j] = np.inf
    return matrix

# merge only the closest contours using the distance matrix
def merge_contours(contours: tuple[cv2.Mat], max_merge_distance=10) -> tuple[cv2.Mat]:
    distance_matrix = get_contour_distance_matrix(contours)
    num_contours = len(contours)
    final_contours = set()
    merged_contours = set()
    contours = list(contours)
    for i in range(num_contours):
        if i in merged_contours:
            continue
        final_contours.add(i)
        while True:
            j = np.argmin(distance_matrix[i])
            if distance_matrix[i, j] < max_merge_distance: # in pixels
                merged_contours.add(i)
                merged_contours.add(j)
                contours[i] = np.concatenate([contours[i], contours[j]], axis=0)
            if distance_matrix[i, j] == np.inf:
                break
            distance_matrix[i, j] = np.inf
    return tuple(cv2.convexHull(contours[i]) for i in final_contours)

def process_contours(contours: tuple[cv2.Mat], max_objects=4, max_merge_distance=10) -> tuple[cv2.Mat]:
    num_objects = len(contours)
    assert num_objects >= max_objects, f"Need at least [{max_objects}] objects but only have [{num_objects}] objects"
    if num_objects > max_objects:
        # try to merge the closest contours
        contours = merge_contours(contours, max_merge_distance)
        num_objects = len(contours)

        cnts = [(i, get_contour_area(c)) for i, c in enumerate(contours)]
        # sort contours by area
        cnts.sort(reverse=True, key=lambda entry : entry[1])
        # take the first max_contours contours
        # no guarantee that after merging the contours, we will have the max_contours
        return tuple(contours[i] for i, _ in cnts[0 : min(len(contours), max_objects)])
    return contours

# score the contours based on their distance from each other and their area
def score_contours(contours: tuple[cv2.Mat], lerp = 0.5) -> float:
    num_contours = len(contours)
    distance_matrix = get_contour_distance_matrix(contours)
    scores = np.zeros(num_contours)
    for i in range(num_contours):
        avg_distance = np.mean([n for n in distance_matrix[i] if n != np.inf])
        scores[i] = lerp * avg_distance + (1 - lerp) * get_contour_area(contours[i])
    return np.sum(scores)

def get_center_points(contours: tuple[cv2.Mat]) -> Points:
    points = np.zeros((len(contours), 2), dtype="int")
    for i, c in enumerate(contours):
        if len(c) == 1:
            points[i] = [c[0, 0, 0], c[0, 0, 1]]
        else:
            M = cv2.moments(c)
            cx = int(M['m10']/(M['m00'] + 0.0001))
            cy = int(M['m01']/(M['m00'] + 0.0001))
            points[i] = [cx, cy]
    return points

def get_extreme_points_from_contour(contour: cv2.Mat) -> tuple[np.ndarray]:
    leftmost = contour[contour[:, :, 0].argmin()][0]
    rightmost = contour[contour[:, :, 0].argmax()][0]
    topmost = contour[contour[:, :, 1].argmin()][0]
    bottommost = contour[contour[:, :, 1].argmax()][0]
    return (leftmost, rightmost, topmost, bottommost)

def get_extreme_points(points: Points) -> Points:
    out = np.zeros((4, 2))
    out[0] = points[0, np.argmin(points[0, 0])] # left most point
    out[1] = points[1, np.argmax(points[0, 0])] # right most point
    out[2] = points[2, np.argmin(points[0, 1])] # top most point
    out[3] = points[3, np.argmax(points[0, 1])] # bottom most point
    return out


def order_center_points(center_points: Points):
    sorted_points = center_points[center_points[:, 0].argsort()]
    left_most = sorted(sorted_points[:2], key=lambda pt : pt[1]) # [top left, bottom left]
    right_most = sorted(sorted_points[2:], key=lambda pt : pt[1]) # [top right, bottom right]
    return np.concatenate([left_most, right_most], axis=0)

def draw_reference_points(img: cv2.Mat, ordered_center_points: Points, from_cameras_perspective=True) -> cv2.Mat:
    top_left = ordered_center_points[0]
    bottom_left = ordered_center_points[1]
    top_right = ordered_center_points[2]
    bottom_right = ordered_center_points[3]
    if not from_cameras_perspective:
        top_left = ordered_center_points[3]
        bottom_left = ordered_center_points[2]
        top_right = ordered_center_points[1]
        bottom_right = ordered_center_points[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1
    image = cv2.putText(img, 'TL', top_left, font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'BL', bottom_left, font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'TR', top_right, font, fontScale, color, thickness, cv2.LINE_AA)
    return cv2.putText(image, 'BR', bottom_right, font, fontScale, color, thickness, cv2.LINE_AA)

def draw_pen_position(img: cv2.Mat, position: np.ndarray) -> cv2.Mat:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1
    return cv2.putText(img, 'Pen', [int(position[0]), int(position[1])], font, fontScale, color, thickness, cv2.LINE_AA)


def get_homography(ordered_src_pts: Points, ordered_dest_pts: Points) -> Matrix:
    assert len(ordered_src_pts) >= 4
    assert len(ordered_dest_pts) >= 4
    assert len(ordered_src_pts) == len(ordered_dest_pts)

    npoints = ordered_src_pts.shape[0]
    A = np.zeros((npoints*2, 8))
    b = np.zeros((npoints*2, 1))
    e = 0

    new_src_points = [
        ordered_src_pts[3], # camera BR maps to paper TL
        ordered_src_pts[2], # camera TR maps to paper BL
        ordered_src_pts[1], # camera BL maps to paper TR
        ordered_src_pts[0], # camera TL maps to paper BR
    ]
    ordered_src_pts = new_src_points
    for i in range(npoints):
        u, v = ordered_src_pts[i]
        u_prime, v_prime = ordered_dest_pts[i]
        A[e] = [u, v, 1, 0, 0, 0, -u_prime * u, -u_prime * v]
        A[e + 1] = [0, 0, 0, u, v, 1, -v_prime * u, -v_prime * v]
        b[e] = u_prime
        b[e + 1] = v_prime
        e += 2

    h = np.linalg.lstsq(A, b, rcond=None)[0]
    h = np.concatenate((np.squeeze(h), [1]), axis=-1)
    return np.reshape(h, (3, 3))


data_dir = os.path.abspath(os.curdir) + "/data/"
#test_image_path = data_dir + "images/resized_test_image.jpg"
test_image_path = data_dir + "images/Test Image w Pen.png"
real_paper_size = (215.9, 279.4) # (width, height) in mm. A4 paper.
landscaped_paper_width, landscaped_paper_height = 1000, 700 # in pixels
# will have to change height and width based on if the paper is landscaped or not
paper_rectangle = np.array([ # landscape orientation
    [0, 0], # top left
    [0, landscaped_paper_height], # bottom left
    [landscaped_paper_width, 0], # top right
    [landscaped_paper_width, landscaped_paper_height], # bottom right
])

def testing():
    test_image = cv2.imread(test_image_path)
    hsv_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
    #threshold = cv2.inRange(hsv_image, (40, 99, 54), (80, 255, 255))
    thresholded_paper = cv2.inRange(hsv_image, (43, 38, 141), (99, 255, 255))
    img_erosion_paper = cv2.erode(thresholded_paper, np.ones((3, 3), np.uint8), iterations=1)

    thresholded_pen = cv2.inRange(hsv_image, (158, 83, 216), (180, 255, 255))
    img_erosion_pen = cv2.erode(thresholded_pen, np.ones((3, 3), np.uint8), iterations=1)

    paper_contours, _ = cv2.findContours(img_erosion_paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pen_contours, _ = cv2.findContours(img_erosion_pen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    paper_contours = process_contours(paper_contours)
    center_points = get_center_points(paper_contours)
    ordered_center_points = order_center_points(center_points)

    pen_contours = process_contours(pen_contours, max_objects=1)
    pen_extreme_points = get_extreme_points_from_contour(pen_contours[0])
    pen_bottom_most_pt = np.concatenate([pen_extreme_points[-1], [1]], axis=-1)

    h = get_homography(ordered_center_points, paper_rectangle)

    pen_position = np.dot(h, pen_bottom_most_pt)
    pen_position /=  pen_position[-1] # divide out w
    # will have to change height and width based on if the paper is landscaped or not
    destination_image = cv2.warpPerspective(test_image, h, (landscaped_paper_width, landscaped_paper_height))

    destination_image = draw_pen_position(destination_image, pen_position)

    #cv2.imshow("Name", draw_reference_points(test_image, ordered_center_points))
    #cv2.imshow("Name", cv2.drawContours(test_image, paper_contours, -1, (0, 255, 0), 3))
    cv2.imshow("Name", destination_image)
    # #cv2.imshow("Name", threshold)
    cv2.waitKey(0)

def main():
    window_width, window_height = 1024, 720
    utility_window_width, utility_window_height = 512, 256

    cap = cv2.VideoCapture(0)
    window_capture_name = "Window Capture"
    cv2.namedWindow(window_capture_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_capture_name, window_width, window_height)

    drawing_capture_name = "Drawing Capture"
    cv2.namedWindow(drawing_capture_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(drawing_capture_name, landscaped_paper_width, landscaped_paper_height)
    drawing_image = np.ones((landscaped_paper_height, landscaped_paper_width, 3))

    pen_position_capture_name = "Pen Position Capture"
    cv2.namedWindow(pen_position_capture_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(pen_position_capture_name, utility_window_width, utility_window_height)

    reference_points_capture_name = "References Points Capture"
    cv2.namedWindow(reference_points_capture_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(reference_points_capture_name, utility_window_width*2, utility_window_height*2)
    
    # calibration settings
    needs_calibrating = True
    last_calibration = time.time()
    time_between_calibrations = 1 # seconds
    ordered_center_points = None
    center_points_score = -np.inf
    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        #frame = cv2.resize(frame, (window_width, window_height)) 
        cv2.imshow(window_capture_name, frame)

        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # threshold and remove noise for the paper
        thresholded_paper = cv2.inRange(hsv_image, (43, 38, 141), (99, 255, 255))
        img_erosion_paper = thresholded_paper
        #img_erosion_paper = cv2.erode(thresholded_paper, np.ones((3, 3), np.uint8), iterations=1)
        #img_erosion_paper = cv2.dilate(img_erosion_paper, np.ones((3, 3), np.uint8), iterations=1)
        # TODO need a calibrate button
        # TODO try dilating after eroding to rejoin any potential contours that may have been seperated
        # due to erosion.
        # threshold and remove noise for the pen
        thresholded_pen = cv2.inRange(hsv_image, (158, 83, 216), (180, 255, 255))
        img_erosion_pen = cv2.erode(thresholded_pen, np.ones((3, 3), np.uint8), iterations=1)

        # shows how the pen is being captured
        cv2.imshow(pen_position_capture_name, img_erosion_pen)

        if (time.time() - last_calibration) > time_between_calibrations:
            print("Needs recalibrating")
            last_calibration = time.time()
            needs_calibrating = True

        paper_contours = get_contours(img_erosion_paper)
        # remove any noisy contours and take the biggest 4 contours
        if needs_calibrating and len(paper_contours) >= 4:
            paper_contours = process_contours(paper_contours)
            # TODO fix score contours
            contour_score = score_contours(paper_contours, lerp=0.3)
            if len(paper_contours) == 4:# and contour_score > center_points_score:
                center_points_score = contour_score
                center_points = get_center_points(paper_contours)
                ordered_center_points = order_center_points(center_points)
                needs_calibrating = False
                cv2.imshow(reference_points_capture_name, draw_reference_points(frame, ordered_center_points, False))
                print("Completed Calibration")

        pen_contours = get_contours(img_erosion_pen)
        if len(pen_contours) >= 1 and ordered_center_points is not None:
            pen_contours = process_contours(pen_contours, max_objects=1)
            pen_extreme_points = get_extreme_points_from_contour(pen_contours[0])
            pen_bottom_most_pt = np.concatenate([pen_extreme_points[-1], [1]], axis=-1)

            h = get_homography(ordered_center_points, paper_rectangle)
            pen_position = np.dot(h, pen_bottom_most_pt)
            pen_position /=  pen_position[-1]
            pen_x, pen_y = pen_position[0], pen_position[1]
            if pen_x >= 0 and pen_x < landscaped_paper_width and pen_y >= 0 and pen_y < landscaped_paper_height:
                #drawing_image = cv2.circle(drawing_image, (int(pen_x), int(pen_y)), 5, (255, 0, 0), 1)
                pen_x, pen_y = int(pen_x), int(pen_y)
                top_left = (max(pen_x - 2, 0), max(pen_y - 2, 0))
                bottom_right = (min(pen_x + 2, landscaped_paper_width - 1), min(pen_y + 2, landscaped_paper_height - 1))            
                drawing_image = cv2.rectangle(drawing_image, top_left, bottom_right, (0, 0, 0), cv2.FILLED)            

        # show drawing image
        cv2.imshow(drawing_capture_name, drawing_image)

        key = cv2.waitKey(30)
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    #testing()
    main()