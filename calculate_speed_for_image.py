
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

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from exif import Image
from datetime import datetime

"""
take for example 5 images per second for 2 seconds
calculate displacement vectors of the images
take another set of x images in x time

calculate the speed of the object in the images

etc
"""


# Constants
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

def load_image(image_path: str):
    """
    Load an image from the specified path.
    """
    image = cv2.imread(image_path, 0)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

def preprocess_image(img: str, sf: float = 1.0):
    """
    Preprocess the image (resize, convert to grayscale, and return in float32 format).
    """
    # Load the image
    image = cv2.imread(img) 
    
    # Check if the image exists
    if image is None:
        raise FileNotFoundError(f"Image not found: {img}")
    
    # Downsample the image
    image_resized = cv2.resize(image, (0, 0), fx=sf, fy=sf)
    
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # Convert the image to float32
    #image_float = np.float32(image_gray)
    
    return image_gray #image_float

def sift_detector(img):
    """
    Detect the SIFT features in the image.
    """
    detector = cv2.SIFT_create()
    kp, des = detector.detectAndCompute(img, None)
    return kp, des

def flann_matching_and_filtering(des1, des2, ratio: float=0.7) -> list:
    """
    Match the features in the images and filter the matches based on the ratio test.
    """
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100) # or pass empty dictionary

    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    # Match descriptors
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    print(f"{len(matches)} matches found.")
    print(f"matches reduced to {len(good_matches)} after ratio ({ratio}) test.")
    
    return good_matches

def apply_ransac(kp1, kp2, g_matches: list, threshold:float=5.0):
    """
    Apply RANSAC to filter matches and compute homography.
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in g_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in g_matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
    matchesMask = mask.ravel().tolist()
    return homography, matchesMask

def visulize_ransac_matches(img1, img2, M, kp1, kp2, matchesMask, matches):
    """
    Returns RANSAC matches in a form that can be inputed into plt.imshow().
    """
    h,w = img1.shape
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)
    return img3

def extract_time_from_exif(image_path: str):
    """
    Extract the time the image was taken from the EXIF data.
    """
    with open(image_path, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time

def calculate_speed(homography, time):
    """
    Calculate the speed of the object in the images.
    """
    # Extract the translation from the homography matrix
    dx = homography[0, 2]
    dy = homography[1, 2]
    
    # Calculate the speed of the object
    speed = np.sqrt(dx**2 + dy**2) / DEPTH_Z
    
    return speed

def pixels_to_meters(dx: float, dy: float, fx: float, fy: float, depth: float) -> tuple:
    """
    Convert the pixel displacement to meters.
    """
    dx_m = (dx / fx) * depth
    dy_m = (dy / fy) * depth
    return dx_m, dy_m

def calculate_velocity(img1_path: str, img2_path: str, time_delta=None) -> float:
    """
    Calculate the velocity of the object in the images.
    """
    # Load and preprocess the images
    # target_float = preprocess_image(img1)
    # reference_float = preprocess_image(img2)
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    
    # Match the features in the images using SIFT and filter the matches using FLANN and Lowe's ratio test
    kp1, des1 = sift_detector(img1)
    kp2, des2 = sift_detector(img2)
    matches = flann_matching_and_filtering(des1, des2)
    
    if len(matches) < MIN_MATCH_COUNT:
        return None
    # Apply RANSAC to filter matches and compute homography
    homography, matches_mask = apply_ransac(kp1, kp2, matches)

    # Visualize the RANSAC matches
    plt.imshow(visulize_ransac_matches(img1, img2, homography, kp1, kp2, matches_mask, matches))

    # Calculate the velocity of the object
    dpx, dpy = homography[0, 2], homography[1, 2]

    dmx, dmy = pixels_to_meters(dpx, dpy, FOCAL_X, FOCAL_Y, DEPTH_Z)

    print(f"dmx: {dmx}, dmy: {dmy}")

    displacement = np.sqrt(dmx**2 + dmy**2)

    print(f"Displacement: {displacement} pixels")

    if time_delta is None:
        time_delta = (extract_time_from_exif(img2_path) - extract_time_from_exif(img1_path)).total_seconds()

    velocity = displacement / time_delta
    
    return velocity/1000


def main():
    # log the time
    time = datetime.now()
    print(f"Time: {time}")
    # open the earth images folder
    velocities = []
    earth_img_folder = os.listdir('earth_img')
    for i in range(len(earth_img_folder) - 1):
        img1 = 'earth_img/' + earth_img_folder[i]
        img2 = 'earth_img/' + earth_img_folder[i+1]
        vel = calculate_velocity(img1, img2)
        if vel is not None:
            velocities.append(vel)
            print(vel)

    # print mean velocity
    time_finish = datetime.now()
    print(f"Time finish: {time_finish}")
    print(f"Time taken: {time_finish - time}")
    print(f"Mean velocity: {np.mean(velocities)} km/s")

    # print(calculate_velocity(TARGET_PATH, REFERENCE_PATH))

main()
