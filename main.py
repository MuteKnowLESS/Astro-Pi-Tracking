import cv2
import numpy as np
from matplotlib import pyplot as plt

TARGET_PATH = 'earth_img\photo_093_53244355532_o.jpg'
REFERENCE_PATH = 'earth_img\photo_092_53245529093_o.jpg'
SCALE_FACTOR = 0.5

# Load the images
target_image = cv2.imread(TARGET_PATH)
reference_image = cv2.imread(REFERENCE_PATH)

# Check if the images exist
if target_image is None or reference_image is None:
    raise FileNotFoundError("One or both image paths are incorrect or the images do not exist.")

# downsample image
target_image_resize = cv2.resize(target_image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
reference_image_resize = cv2.resize(reference_image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

# Convert the images to grayscale
target_image_gray = cv2.cvtColor(target_image_resize, cv2.COLOR_BGR2GRAY)
reference_image_gray = cv2.cvtColor(reference_image_resize, cv2.COLOR_BGR2GRAY)

# convert the images to float
target_float = np.float32(target_image_gray)
reference_float = np.float32(reference_image_gray)

def preprocess_image(image_path: str, scale_factor: float = 0.5):
    """
    Function to preprocess the image
    """
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image exists
    if image is None:
        raise FileNotFoundError("The image path is incorrect or the image does not exist.")

    # Downsample the image
    image_resize = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)

    # Convert the image to float
    image_float = np.float32(image_gray)

    return image_float

def harris_corner(target_img_float, ref_img_float):
    """
    Function to detect the Harris corners in the target image
    """
    # Detect the corners in the images
    # The parameters are the image, block size, ksize, and k
    # The block size is the size of the neighborhood considered for corner detection
    # The ksize is the aperture parameter for the Sobel operator
    # The k is the Harris detector free parameter in the equation
    corners_target = cv2.cornerHarris(target_img_float, 2, 3, 0.04)
    corners_ref = cv2.cornerHarris(ref_img_float, 2, 3, 0.04)
    corners_target = cv2.dilate(corners_target, None)
    corners_ref = cv2.dilate(corners_ref, None)

    # Get Keypoints from Corners detected by Harris-Corner Detection
    # The keypoints are the x and y coordinates of the corners
    # The size of the keypoints is set to 20
    # The threshold is set to 0.01 * the maximum value of the corners
    # The keypoints are stored in a list as cv2.KeyPoint objects
    keypoints_target = [cv2.KeyPoint(float(x[1]), float(x[0]), 20) for x in np.argwhere(corners_target > 0.01 * corners_target.max())]
    keypoints_reference = [cv2.KeyPoint(float(x[1]), float(x[0]), 20) for x in np.argwhere(corners_ref > 0.01 * corners_ref.max())]

    return keypoints_target, keypoints_reference

kp_target, kp_reference = harris_corner(target_float, reference_float)

#ORB Feature Matching

# orb = cv2.ORB_create()
# keypoints_target, descriptors_target = orb.compute(target_image_gray, kp_target)
# keypoints_reference, descriptors_reference = orb.compute(reference_image_gray, kp_reference)

def SIFT_feature_matching(target_img, reference_img, kp_target, kp_reference):
    """
    Function to match the features using SIFT
    """
    # Create a SIFT object
    sift = cv2.SIFT_create()
    keypoints_target, descriptors_target = sift.compute(target_img, kp_target)
    keypoints_reference, descriptors_reference = sift.compute(reference_img, kp_reference)

    # Create a Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match the descriptors
    matches = bf.match(descriptors_target, descriptors_reference)
    matches = sorted(matches, key = lambda x:x.distance)

    return matches

def ORB_feature_matching(target_img, reference_img, kp_target, kp_reference):
    """
    Function to match the features using ORB
    """
    # Create an ORB object
    orb = cv2.ORB_create()
    keypoints_target, descriptors_target = orb.compute(target_img, kp_target)
    keypoints_reference, descriptors_reference = orb.compute(reference_img, kp_reference)

    # Create a Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the descriptors
    matches = bf.match(descriptors_target, descriptors_reference)
    matches = sorted(matches, key = lambda x:x.distance)

    return matches

# def BF_feature_matching(target_img, reference_img, kp_target, kp_reference):
#     """
#     Function to match the features using Brute Force
#     """
#     # Create a Brute Force Matcher
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

#     # Match the descriptors
#     matches = bf.match(descriptors_target, descriptors_reference)
#     matches = sorted(matches, key = lambda x:x.distance)

    return matches

def RANSAC(matches, keypoints_target, keypoints_reference):
    """
    Function to apply RANSAC to the matches
    """
    # Apply RANSAC to the matches
    # The RANSAC function takes the keypoints, the matches, and the maximum distance between the keypoints
    # The function returns the homography matrix and the mask
    homography, mask = cv2.findHomography(np.float32([keypoints_target[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2),
                                            np.float32([keypoints_reference[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2),
                                            cv2.RANSAC, 5.0)

    return homography, mask

def draw_matches(target_img, reference_img, keypoints_target, keypoints_reference, matches):
    """
    Function to draw the matches
    """
    target_matches = cv2.drawMatches(target_img, keypoints_target, reference_img, keypoints_reference, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return target_matches

def draw_keypoints(target_matches_list: list):
    # display muliple matches together
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    axs[0].imshow(target_matches_list[0])
    axs[0].set_title('SIFT Matches')
    axs[1].imshow(target_matches_list[1])
    axs[1].set_title('ORB Matches')
    plt.show()


def main():

    # Match the features using SIFT, ORB, and SURF

    sift_matches = SIFT_feature_matching(target_image_gray, reference_image_gray, kp_target, kp_reference)

    orb_matches = ORB_feature_matching(target_image_gray, reference_image_gray, kp_target, kp_reference)

    

    # Apply RANSAC to the matches    

    orb_homography, orb_mask = RANSAC(orb_matches, kp_target, kp_reference)

    sift_homography, sift_mask = RANSAC(sift_matches, kp_target, kp_reference)

    # Draw the matches
    sift_target_matches = draw_matches(target_image_gray, reference_image_gray, kp_target, kp_reference, sift_matches)

    orb_target_matches = draw_matches(target_image_gray, reference_image_gray, kp_target, kp_reference, orb_matches)

    # print the x and y translation
    print("SIFT: ", sift_homography[0, 2], sift_homography[1, 2], "matches: ", len(sift_matches))
    print("ORB: ", orb_homography[0, 2], orb_homography[1, 2], "matches: ", len(orb_matches))

    # Display the matches
    target_matches_list = [sift_target_matches, orb_target_matches]
    draw_keypoints(target_matches_list)

if __name__ == '__main__':
    main()