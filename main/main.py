
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

### Camera Intrinsic parameters

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
from logzero import logger, logfile
from scipy.stats import mode
from scipy.stats import trim_mean 
import time
import csv
import sys
from picamzero import Camera

"""
take 10 images 5 seconds apart. process the images in pairs to calculate the speed of the ISS. repeat this process until reach time threshhold. save the results to a text file.
"""

# Constants

PROGRAM_START_TIME = datetime.now()
PROGRAM_TIMEOUT = 600 # 10 minutes

# TARGET_PATH = 'earth_img/photo_091_53245728575_o.jpg' # this is the first image taken in order of the pair
# REFERENCE_PATH = 'earth_img/photo_092_53245529093_o.jpg' # this is the second image taken in order of the pair

# ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

LOG_PATH = 'tmp.log'
IMG_PATH = 'images/'
CSV_DATA_PATH = 'data.csv'
RESULTS_PATH = 'result.txt'

SCALE_FACTOR = 1
MIN_MATCH_COUNT = 10

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
    """Load an image from the specified path."""
    try:
        image = cv2.imread(image_path, 0)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")


def sift_detector(img):
    """Detect the SIFT features in the image."""
    detector = cv2.SIFT_create()
    kp, des = detector.detectAndCompute(img, None)
    return kp, des


def flann_matching_and_filtering(des1, des2, ratio: float=0.7) -> list:
    """Match the features in the images and filter the matches with lowe's ratio test."""
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

    logger.info(f"{len(matches)} matches found. matches reduced to {len(good_matches)} after ratio ({ratio}) test.")
    
    return good_matches


def apply_ransac(kp1, kp2, matches: list, threshold: float=5.0):
    """Compute the homography matrix using RANSAC."""
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
    matchesMask = mask.ravel().tolist()
    return homography, matchesMask


def visulize_inlier_matches(img1, img2, M, kp1, kp2, matchesMask, matches):
    """Returns RANSAC matches in a form that can be inputed into plt.imshow()."""
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
        logger.info(f"Time extracted from EXIF: {time}")
    return time


def pixels_to_meters(pixel_dx: float, pixel_dy: float, k_fx: float, k_fy: float, depth_m: float) -> tuple:
    """Convert the pixel displacement to meters."""
    dx_m = (pixel_dx / k_fx) * depth_m
    dy_m = (pixel_dy / k_fy) * depth_m
    return dx_m, dy_m


def calculate_velocity(img1_path: str, img2_path: str, time_delta=None, plot_matches: bool=False) -> float:
    """Calculate the velocity of the camera relative to nadir images of a ground plane (earth)."""

    # Load images
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    
    # Detect SIFT features.
    kp1, des1 = sift_detector(img1)
    kp2, des2 = sift_detector(img2)

    # Match features and filter with Lowe's ratio test
    matches = flann_matching_and_filtering(des1, des2)
    if len(matches) < MIN_MATCH_COUNT:
        logger.error(f"Insufficient matches found: {len(matches)} for {img1_path} and {img2_path}")
        return None
    
    # Apply RANSAC to for inlier mask and compute homography
    H, matches_mask = apply_ransac(kp1, kp2, matches)
    logger.info(f"RANSAC homography matrix: {H}")

    # defines the displacement vector in pixels from the homography matrix
    dpx, dpy = H[0, 2], H[1, 2]
    dx_m, dy_m = pixels_to_meters(dpx, dpy, FOCAL_X, FOCAL_Y, DEPTH_Z)
    displacement_magnitude = np.sqrt(dx_m**2 + dy_m**2)
    logger.info(f"Pixel displacement vector: {dpx, dpy}")
    logger.info(f"physical displacement vector: {dx_m, dy_m} (meters)")
    logger.info(f"physical displacement magnitude: {displacement_magnitude} (meters)")

    # calculate the time delta if not provided
    if time_delta is None:
        logger.warning("Time delta not provided. Using EXIF data to calculate time delta.")
        time_delta = (extract_time_from_exif(img2_path) - extract_time_from_exif(img1_path)).total_seconds()
    # check if time delta is datetime object tuple (time1, time2)
    if isinstance(time_delta, tuple):
        logger.warning("Time delta is a datetime tuple. Calculating time delta from datetime tuple.")
        time_delta = (time_delta[1] - time_delta[0]).total_seconds()
    
    logger.info(f"Time delta: {time_delta} seconds")

    # calculate the velocity
    velocity_mps = displacement_magnitude / time_delta
    velocity_kps = velocity_mps / 1000.0
    #logger.info(f"Velocity: {velocity_kps} km/s")

     # Visualize the inlier matches. Comment out if not needed.
    if plot_matches: 
        pass
    #     logger.info("Visualizing matches.")
    #     plt.imshow(visulize_inlier_matches(img1, img2, H, kp1, kp2, matches_mask, matches))
    #     # add time, pixel, physical displacement and velocity to the plot as subtitles (rounded to 5 decimal places)
    #     plt.suptitle(f"Time Delta (s): {(round(time_delta, 5))} \nPixel displacement (px): ({round(dpx, 5)}, {round(dpy, 5)})\nPhysical displacement (m): ({round(dx_m, 5)}, {round(dy_m, 5)})\nVelocity (km/s): {round(velocity_kps, 5)} km/s")
    #     plt.show()

    return velocity_kps

def calculate_mean_velocity(velocities: list) -> float:
    """Calculate the mean velocity from a list of velocity estimates."""
    if len(velocities) == 0:
        logger.error("No velocity estimates found.")
        return 0
    mean_velocity = np.mean(velocities)
    return mean_velocity


def calculate_median_velocity(velocities: list) -> float:
    """Calculate the median velocity from a list of velocity estimates."""
    if len(velocities) == 0:
        logger.error("No velocity estimates found.")
        return 0
    median_velocity = np.median(velocities)
    return median_velocity


def calculate_mode_velocity(velocities: list) -> float:
    """Calculate the mode velocity from a list of velocity estimates."""
    if len(velocities) == 0:
        logger.error("No velocity estimates found.")
        return 0
    mode_velocity = mode(velocities, keepdims=True).mode[0]
    return mode_velocity


def calculate_IQR_mean_velocity(velocities: list, mid_perc: float=50) -> float:
    """Calculate the mean velocity from a list of velocity estimates after removing outliers."""
    if len(velocities) == 0:
        logger.error("No velocity estimates found.")
        return 0
    lb = mid_perc/2
    ub = 100 - lb
    q1, q3 = np.percentile(velocities, [lb, ub])
    iqr_velocities = [v for v in velocities if q1 <= v <= q3]
    iqr_mean_velocity = np.mean(iqr_velocities)
    return iqr_mean_velocity

def calculate_trimmed_mean_velocity(velocities: list, trim: float=0.1) -> float:
    """Calculate the trimmed mean velocity from a list of velocity estimates."""
    velocities_trimmed_mean = trim_mean(velocities, trim)
    return velocities_trimmed_mean

def calculate_velocity_from_images(image_folder: str, time_sleep: int, num_images: int) -> float:
    """Calculate the velocity of the ISS from a series of images."""
    velocities = []
    for i in range(num_images - 1):
        img1 = f"{image_folder}/photo_{i}.jpg"
        img2 = f"{image_folder}/photo_{i+1}.jpg"
        try:
            vel = calculate_velocity(img1, img2)
            velocities.append(vel)
            logger.info(f"Velocity: {vel} km/s")
        except Exception as e:
            logger.error(f"Error calculating velocity for {img1} and {img2}: {e}")
    return velocities


def save_velocity_to_csv(velocity: float, filename: str = CSV_DATA_PATH, src12: tuple=('','')) -> None:
    """Appends a single velocity to the CSV file."""
    try:
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().isoformat(), velocity, src12[0], src12[1]])
        logger.info(f"Velocity {velocity} km/s appended to {filename}")
    except Exception as e:
        logger.error(f"Error appending velocity {velocity} to {filename}: {e}")


def save_velocities_to_csv(velocities: list[float],filename: str = CSV_DATA_PATH, src12="") -> None:
    """Appends a list of velocities to the CSV file."""
    try:
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            for x, vel in enumerate(velocities):
                if isinstance(vel, (int, float)):  # Ensure valid data type
                    if src12 == "":
                        writer.writerow([datetime.now().isoformat(), vel])
                    else:
                        writer.writerow([datetime.now().isoformat(), vel, src12[x][0], src12[x][1]])
                else:
                    logger.warning(f"Skipping invalid velocity {vel} (not a number)")
        logger.info(f"{len(velocities)} velocities appended to {filename}")
    except Exception as e:
        logger.error(f"Error appending velocities to {filename}: {e}")


def load_velocities_from_csv(filename: str = CSV_DATA_PATH) -> list[tuple[str, float]]:
    """Reads velocities from the CSV file with error handling."""
    velocities: list[tuple[str, float]] = []
    
    try:
        with open(filename, mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 2:  # Ensure there are exactly two columns (timestamp and velocity)
                    try:
                        timestamp, velocity = row
                        velocity = float(velocity)  # Ensure velocity is a float
                        velocities.append((timestamp, velocity))
                    except ValueError:
                        logger.warning(f"Skipping invalid row due to value error: {row}")
                    except Exception as e:
                        logger.error(f"Error processing row {row}: {e}")
                else:
                    logger.warning(f"Skipping row with unexpected number of columns: {row}")
    
    except FileNotFoundError:
        logger.error(f"File {filename} not found.")
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}")
    
    return velocities

def save_velocity_to_txt(velocity: float, filename: str = 'result.txt') -> None:
    """Save the velocity to a text file."""
    try:
        with open(filename, 'a') as f:
            f.write(f"{velocity:.5g} km/s")
        #logger.info(f"Velocity {velocity:.5g} km/s saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving velocity {velocity:.5g} to {filename}: {e}")
    
def save_velocities_to_txt(velocities: list, filename: str = 'result.txt', head: str="") -> None:
    """Save the velocities to a text file."""
    try:
        with open(filename, 'a') as f:
            f.write(head)
            for velocity in velocities:
                f.write(f'{velocity}\n')
            f.write('\n')
        logger.info(f"Velocities saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving velocities to {filename}: {e}")

def get_time_from_filename(filename: str) -> datetime:
    """Extract the time from the filename. Assumes the filename is formatted as: "img_YYYYmmdd_HHMMSS_mmm.jpeg" where mmm are the milliseconds."""
    try:
        # Remove the "img_" prefix and file extension
        timestamp_str = filename.replace("img_", "").split(".")[0]  # e.g. "20230430_073756_123"
        parts = timestamp_str.split('_')
        if len(parts) != 3:
            raise ValueError(f"Invalid filename format: {filename}")
        # append "000" to convert milliseconds to microseconds
        ts_full = f"{parts[0]}_{parts[1]}_{parts[2]}000"
        dt = datetime.strptime(ts_full, "%Y%m%d_%H%M%S_%f")
        return dt
    except Exception as e:
        raise ValueError(f"Error extracting time from filename: {e}")
    
def get_time_delta(img1_path: str, img2_path: str, t_threshold:float=2.0) -> float:
    """Calculate the time delta between two images from the filenames. uses exif data if the time delta is less than t_threshold"""
    # try to extract the time from the filename
    try:
        time1_filename = get_time_from_filename(img1_path)
        time2_filename = get_time_from_filename(img2_path)
        delta_filename = (time2_filename - time1_filename).total_seconds()
    except Exception as e:
        delta_filename = None
        #logger.warning(f"Failed to extract time from filename: {e}")

    # Get time delta from EXIF:
    try:
        time1_exif = extract_time_from_exif(img1_path)
        time2_exif = extract_time_from_exif(img2_path)
        delta_exif = (time2_exif - time1_exif).total_seconds()
    except Exception as e:
        delta_exif = None
        #logger.warning(f"Failed to extract time from EXIF: {e}")

    # Decide which delta to use:
    if delta_filename is None and delta_exif is None:
        raise ValueError("Failed to extract time from both filename and EXIF data.")
    if delta_filename is None:
        return delta_exif
    if delta_exif is None:
        return delta_filename
        # If the two deltas differ by more than the threshold, log a warning and use EXIF delta.
    if abs(delta_filename - delta_exif) > t_threshold:
        logger.warning(f"Time delta from filename ({delta_filename}s) and EXIF ({delta_exif}s) differ by more than {threshold} seconds; using EXIF data.")
        return delta_exif
    else:
        return delta_filename
    
def capture_images(cam, n_images: int=10, interval: float=5.0, save_path: str='img_sets/') -> None:
    """Capture a series of images at a specified interval. returns a list of saved image paths."""

    saved_files = []

    for i in range(n_images):
        t_start = time.monotonic()
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] # Timestamp with ms precision
        img_path = f"{save_path}photo_{time_stamp}.jpg"
        cam.capture_image(img_path)
        saved_files.append(img_path)
        logger.info(f"Image {i} captured at {time_stamp}")
        capture_time = time.monotonic()
        elapsed_time = capture_time - t_start
        time.sleep(max(0, interval - elapsed_time)) # Ensure the interval is maintained

    return saved_files

def process_img_set(img_paths: list) -> None:
    """Process a set of images and calculate the velocity of the ISS."""
    velocities = []
    imgs_src = []
    for i in range(len(img_paths) - 1):
        logger.info(f"Processing images {img_paths[i]} and {img_paths[i+1]}")
        img1 = img_paths[i]
        img2 = img_paths[i+1]
        # Calculate the time delta between the images from file name ("%Y%m%d_%H%M%S_%f")[:-3]
        try:
            time_delta = get_time_delta(img1, img2, t_threshold=2.0)
        except Exception as e:
            logger.error(f"Error calculating time delta for {img1} and {img2}: {e}")
            #time_delta = 5.0 # Default to 5 seconds if time delta cannot be calculated
            continue
        logger.info(f"Time delta: {time_delta} seconds")

        if time_delta < 0:
            logger.warning(f"Negative time delta detected for {img1} and {img2}.")
        vel = calculate_velocity(img1, img2, time_delta)
        if vel is not None:
            if vel < 0:
                    logger.warning(f"Negative velocity detected for {img1} and {img2}.")
            if vel == 0:
                    logger.warning(f"Zero velocity detected for {img1} and {img2}.")
            velocities.append(np.abs(vel))
            imgs_src.append((img1, img2))
            logger.info(f"Velocity: {vel} km/s")

    return velocities, imgs_src # Return the velocities and image sources as lists


def image_capture_and_processing(cam, n_images: int=10, interval: float=5.0, save_path: str='img_sets/') -> None:
    """Capture a series of images at a specified interval and process them. Save the results to csv file"""
    start_time = datetime.now()
    img_paths = capture_images(cam, n_images, interval, save_path)
    vels, img_src = process_img_set(img_paths)
    save_velocities_to_csv(vels, src12=img_src, filename=CSV_DATA_PATH)
    
    #velocities = [v[1] for v in load_velocities_from_csv()]
    mean_velocity = calculate_mean_velocity(vels)
    median_velocity = calculate_median_velocity(vels)
    mode_velocity = calculate_mode_velocity(vels)
    iqr_mean_velocity = calculate_IQR_mean_velocity(vels)
    trimmed_mean_velocity = calculate_trimmed_mean_velocity(vels)

    # delete every other image to save space without using the os module
    # for i in range(0, len(img_paths), 2):
    #     try:
    #         os.remove(img_paths[i])
    #     except Exception as e:
    #         logger.error(f"Error deleting image {img_paths[i]}: {e}")


  
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    avg_vels = [f"mean velocity: {mean_velocity}", f"median velocity: {median_velocity}", f"mode velocity: {mode_velocity}", f"IQR mean velocity: {iqr_mean_velocity}", f"trimmed mean velocity: {trimmed_mean_velocity}"]
    save_velocities_to_txt(avg_vels, filename=RESULTS_PATH, head=f"Compute Time: {elapsed_time.total_seconds()}, Folder: {save_path}\n")

    
    logger.info(f"time taken to complete cycle: {elapsed_time}")
    return elapsed_time


def main():

    # setup the camera
    cam = Camera()
    #cam.greyscale() = True
    cam.setup()
    time_start = datetime.now()
    image_capture_and_processing(cam, n_images=10, interval=5.0, save_path='img_sets/')    # save a rolling average of the velocities from the csv file to a text file
    rolling_velocities = [v[1] for v in load_velocities_from_csv(filename=CSV_DATA_PATH)]
    rolling_iqr_mean_velocity = calculate_IQR_mean_velocity(rolling_velocities, mid_perc=50)
    time_finish = datetime.now()
    logger.info(f"Time to complete: {time_finish - time_start}")
    remaining_time = PROGRAM_TIMEOUT - (time_finish - PROGRAM_START_TIME).total_seconds()
    logger.info(f"Time remaining: {remaining_time}")
    logger.info(f"Rolling IQR mean velocity: {rolling_iqr_mean_velocity} km/s")
    return


if __name__ == "__main__":
    main()