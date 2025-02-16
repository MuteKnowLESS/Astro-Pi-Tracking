

"""
Outline:
- program needs to produce a numeric output of the average speed that the ISS is travelling. This is also known as the magnitude of velocity.
- The output that your program produces must be a .txt file as described in the Mission Space Lab creator guide (rpf.io/msl-creators).
- The numeric output should use no more than 5 significant numbers (5 digits in total including decimal places, for example, 1.2345 km/s).
- The speed must be given in kilometres per second (km/s). 

Task: 
- measure the average linear speed that the ISS is travelling around the Earth (not how much the ISS is rotating). 

General Requirements:
- program does not rely on interaction with an astronaut
- program is written in Python 3.11 and is named main.py, and it runs without errors when tested with the Astro Pi Replay Tool using python3 main.py.
- program does not rely on any additional libraries other than those listed in the Mission Space Lab creator guide (rpf.io/msl-creators).
- program monitors its running time and stops after 10 minutes have elapsed.
- program is not allowed to retain more than 42 images at the end of the 10 minutes — though it can store more than that while it is running.
- zipped program must not be more than 3MB, unless it includes a TensorFlow Lite (.tflite) machine learning model, in which case your zipped program must not be more than 7MB.

Security Requirements:
- program is well documented and easy to understand, and there is no attempt to hide or obfuscate what a piece of code does.
- program does not start a system process, or run another program or any command usually entered on the terminal (e.g. vcgencmd).
- program does not use networking.
- program does not include malicious code

Files and Threads Requirements:
- program does not use threads, or if it does, it does so only by using the threading library; threads are managed carefully and closed cleanly, and their use is clearly explained through comments in the code.
- program only saves data in the same folder where your main Python file is, as described in the Mission Space Lab creator guide (i.e. using the special __file__ variable); program does not attempt to create new directories for storing data, and no absolute path names are used.
- program runs without errors and does not raise any unhandled exceptions.
- Any files that your program creates have names that only include letters, numbers, dots (.), dashes (-), or underscores (_).
- program does not use more than 250MB of space to store data.
- As well as containing main.py file, the zip file that you submit must only contain the following file types: .py, .main, .csv, .txt, .jpg, .png, .yuv, .json, .toml, .yaml, .tflite.
- In addition to result.txt file, the output of your program must only include the following file types: .csv, .txt, .log, .jpg, .png, .yuv, .raw (camera), .h264, .json, .toml, .yaml.

"""


"""
### our approach:
- use images to determine the speed of the ISS

### method:
- use picamera to take images of the earth and save the delta time between the images
- once images are captured, use openCV to process the images in pairs


- use openCV sift detector to detect features in the images
- use flann matcher to match the features in the images and Lowe's ratio test to filter the matches
- use RANSAC to filter the matches and compute homography
- use the homography to find the transformation matrix
(https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html)

- use the transformation matrix to find the displacement of the ISS
- use the displacement and delta time to find the speed of the ISS
- store each paired speed estimate in a list
- calculate the average speed of the ISS from the list of speed estimates
- output the average speed to a text file
"""


"""
### Constants
TARGET_PATH & REFERENCE_PATH for testing
SCALE_FACTOR to resize the images if performance is an issue
MIN_MATCH_COUNT to filter the matches for RANSAC and homography

### Hardware
- Raspberry Pi 4 Model B
- Raspberry Pi High Quality Camera (https://www.sony-semicon.com/files/62/pdf/p-13_IMX477-AACK_Flyer.pdf)
    - 12.3 megapixel Sony IMX477 sensor
    - 7.857mm diagonal image size
    - 1.55 μm x 1.55 μm pixel size
    - Chip size 7.564 mm (H) x 5.476 mm (V)
    - m12 mount?
        - Back focus length of lens: 2.6mm-11.8mm
    - 5mm fixed focal length lens
    - Full (4:3) resolution: 4056 x 3040
- MidOpt red filter
    - ???


### Camera Intrinsic parameters
HQ Camera
> The Raspberry Pi High Quality (HQ) Camera from Raspberry Pi offers a 12-megapixel resolution. It comes with a 5mm fixed focal length lens, designed for a wide field of view to capture more of the target area. The lens features low distortion and high resolution, perfect for detailed imaging applications.
(https://astro-pi.org/about/the-sensors)

https://stackoverflow.com/questions/78072261/how-to-find-cameras-intrinsic-matrix-from-focal-length

focal length [pixels] =
    focal length [mm] / sensor pixel size [µm/pixels]

sensor pixel size [µm/pixels] =
    sensor size along one edge [mm or µm] / pixels along that edge [pixels]

7.9 mm / 
    
focal length [pixels] =
    5 mm / 1.55 µm = 3226 pixels

f_mm = 5
sensor_width = 7.564
sensor_height = 5.476
image_width = 4056
image_height = 3040

f_x = f_mm * (image_width / sensor_width)
f_y = f_mm * (image_height / sensor_height)

f_x = 5 mm * (4056 / 7.564 mm) = 2681.1210 pixels
f_y = 5 mm * (3040 / 5.476 mm) = 2775.7487 pixels

c_x = image_width / 2 = 2028 pixels
c_y = image_height / 2 = 1520 pixels

k = [f_x, 0, c_x,]
    [0, f_y, c_y,]
    [0,  0 ,  1  ]

"""


# =====================================IMPORTS=================================
import numpy as np
import cv2 
from matplotlib import pyplot as plt
from exif import Image
from datetime import datetime


# =====================================CONSTANTS===============================
TARGET_PATH = 'earth_img/photo_091_53245728575_o.jpg' #'earth_img/photo_088_53244355572_o.jpg' # #'earth_img/photo_092_53245529093_o.jpg' # this is the first image taken in order of the pair
REFERENCE_PATH = 'earth_img/photo_092_53245529093_o.jpg'# 'earth_img/photo_089_53245235151_o.jpg' # #'earth_img/photo_093_53244355532_o.jpg' # this is the second image taken in order of the pair
SCALE_FACTOR = 1
MIN_MATCH_COUNT = 10

#earth_img/photo_088_53244355572_o.jpg
#earth_img/photo_089_53245235151_o.jpg

FULL_FRAME_WIDTH = 4056 # pixels
FULL_FRAME_HEIGHT = 3040 # pixels
SENSOR_HIEGHT = 5.476 # mm
SENSOR_WIDTH = 7.564 # mm
FOCAL_LENGTH = 5 # mm

FOCAL_X = FOCAL_LENGTH * (FULL_FRAME_WIDTH / SENSOR_WIDTH)
FOCAL_Y = FOCAL_LENGTH * (FULL_FRAME_HEIGHT / SENSOR_HIEGHT)
PRINCIPLE_POINT_X = FULL_FRAME_WIDTH / 2
PRINCIPLE_POINT_Y = FULL_FRAME_HEIGHT / 2

CAMERA_K = np.array([[FOCAL_X, 0, PRINCIPLE_POINT_X], 
                     [0, FOCAL_Y, PRINCIPLE_POINT_Y], 
                     [0, 0, 1]])

DEPTH_Z = 420000 #(4.2e+05) meters


# =====================================FUNCTIONS===============================
def preprocess_image(image_path: str, scale_factor: float = 0.5):
    """
    Load and preprocess an image (resize, convert to grayscale, and return in uint8 format).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    return image_gray  # Keep as uint8

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the specified path.
    """
    image = cv2.imread(image_path, 0)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

def sift_detect(image: np.ndarray):
    """
    Detect SIFT features in the image
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    return kp, des

def flann_match(des1, des2):
    """
    Use FLANN matcher to match the features in the images
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches

def filter_matches(matches, ratio=0.7):
    """
    Filter the matches using Lowe's ratio test
    """
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def apply_ransac(matches, kp1, kp2):
    """
    Apply RANSAC to filter matches and compute homography.
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()
    return M, mask

def eliminate_outliers(matches, mask):
    """
    Eliminate outliers from the matches using the RANSAC mask
    """
    mask = mask.ravel().tolist()
    inliers = [m for i, m in enumerate(matches) if mask[i] == 1]
    return inliers

def calculate_displacement_homography(homography):
    """
    Calculate the displacement using the homography matrix
    """
    displacement = np.array([homography[0, 2], homography[1, 2]])
    # this fuction is stupid and redundant
    return displacement

def calculate_mean_displacement_matches(matches, kp1, kp2):
    """
    Calculate the displacement using the mean of the matched keypoints
    """
    displacement = np.mean([np.array(kp2[m.trainIdx].pt) - np.array(kp1[m.queryIdx].pt) for m in matches], axis=0)
    return displacement

def visualize_matches(image1, image2, kp1, kp2, matches):
    """
    Visualize the matches between two images
    """
    return cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def show_image(image):
    """
    Show the matches between two images
    """
    plt.imshow(image)
    plt.show()

def calculate_speed(displacement, delta_time):
    """
    Calculate the speed using the displacement and delta time
    """
    #magnitude = np.linalg.norm(displacement)
    
    speed = np.linalg.norm(displacement) / delta_time
    return speed

def extract_timestamp(image_path):
    """
    Extract the timestamp from the image filename
    """
    with open(image_path, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time

def calculate_delta_time(time1, time2):
    """
    Calculate the time difference between two timestamps
    """
    delta_time = (time2 - time1).total_seconds()
    return delta_time

def pixel_to_meters(dx, dy, f_x, f_y, z):
    """
    Convert pixel displacement to meters
    """
    # dx_m = (dx - K[0,2]) * z / K[0,0]
    # dy_m = (dy - K[1,2]) * z / K[1,1]
    dx_m = (dx * z) / f_x
    dy_m = (dy * z) / f_y
    return dx_m, dy_m

def calculate_average_speed(speeds):
    """
    Calculate the average speed from a list of speed estimates
    """
    average_speed = np.mean(speeds)
    return average_speed

def main():
    # img1 = preprocess_image(TARGET_PATH, SCALE_FACTOR)
    # img2 = preprocess_image(REFERENCE_PATH, SCALE_FACTOR)
    #img1 = load_image(TARGET_PATH)
    #img2 = load_image(REFERENCE_PATH)
    img1 = cv2.imread('earth_img\photo_091_53245728575_o.jpg',0)          # queryImage
    img2 = cv2.imread('earth_img\photo_092_53245529093_o.jpg',0) # trainImage

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # kp1, des1 = sift_detect(img1)
    # kp2, des2 = sift_detect(img2)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    print(len(good))
    print(len(matches))

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img3),plt.show()

    #matches = flann_match(des1, des2)

    #print("matches: ", len(matches))
    #matches = filter_matches(matches, 0.7)

    

    homography, mask = apply_ransac(matches, kp1, kp2)
    inliers_matches = eliminate_outliers(matches, mask)

    print("inliers: ", len(inliers_matches))  

    matchesMask = mask.ravel().tolist()

    print("matches mask: ", len(matchesMask))

    h,w = img1.shape
    #print("h, w: ", h,w)
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #print("pst: ", pts)
    dst = cv2.perspectiveTransform(pts,homography)
    print("dts: ", dst)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()


    # displacement_homo = calculate_displacement_homography(homography)
    displacement_mean = calculate_mean_displacement_matches(matches, kp1, kp2)

    print(f"Mean Displacement Vector: dx={displacement_mean[0]}, dy={displacement_mean[1]}")

    print(f"Homography Matrix Displacement: dx={homography[0,2]}, dy={homography[1,2]}")

    displacement_homo_meters = pixel_to_meters(homography[0,2], homography[1,2], FOCAL_X, FOCAL_Y, DEPTH_Z)
    displacement_mean_meters = pixel_to_meters(displacement_mean[0], displacement_mean[1], FOCAL_X, FOCAL_Y, DEPTH_Z)

    

    print(displacement_homo_meters)
    print(displacement_mean_meters)

    #show_image(visualize_matches(img1, img2, kp1, kp2, matches))

    time1 = extract_timestamp(TARGET_PATH)
    time2 = extract_timestamp(REFERENCE_PATH)

    delta_time = time2-time1
    print(delta_time)
    # speed_homo = calculate_speed(displacement_homo_meters, delta_time.total_seconds()) # m/s
    # speed_mean = calculate_speed(displacement_mean_meters, delta_time.total_seconds()) # m/s

    displacement_homo_km = np.sqrt(displacement_homo_meters[0]**2+displacement_homo_meters[1]**2)/1000

    displacement_mean_km = np.sqrt(displacement_mean_meters[0]**2+displacement_mean_meters[1]**2)/1000


    print(displacement_mean_km)
    print(displacement_homo_km)
    
    
    print("mean speed (km/s): ", displacement_mean_km/delta_time.total_seconds())
    print("homo speed (km/s): ", displacement_homo_km/delta_time.total_seconds())
   

    # speed = calculate_speed(displacement, delta_time)


main()


