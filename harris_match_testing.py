import cv2
import numpy as np
from matplotlib import pyplot as plt

# Constants
TARGET_PATH = 'earth_img/photo_092_53245529093_o.jpg'
REFERENCE_PATH = 'earth_img/photo_093_53244355532_o.jpg'
SCALE_FACTOR = 0.5

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


def detect_harris_corners(image: np.ndarray):
    """
    Detect Harris corners and return them as keypoints.
    """
    corners = cv2.cornerHarris(image, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 20) for x in np.argwhere(corners > 0.01 * corners.max())]
    return keypoints


def feature_matching(method, target_img, reference_img, kp_target, kp_reference):
    if method == 'SIFT':
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    elif method == 'ORB':
        detector = cv2.ORB_create()
        norm_type = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Invalid method: {method}")
    
    kp_target, des_target = detector.compute(target_img, kp_target)
    kp_reference, des_reference = detector.compute(reference_img, kp_reference)
    if des_target is None or des_reference is None:
        raise ValueError("No descriptors found in one or both images.")
    
    matcher = cv2.BFMatcher(norm_type, crossCheck=True)
    matches = matcher.match(des_target, des_reference)
    return sorted(matches, key=lambda x: x.distance)

def apply_ransac(matches, kp_target, kp_reference):
    """
    Apply RANSAC to filter matches and compute homography.
    """
    src_pts = np.float32([kp_target[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_reference[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return homography, mask

def draw_matches(target_img, reference_img, kp_target, kp_reference, matches):
    """
    Draw feature matches.
    """
    return cv2.drawMatches(target_img, kp_target, reference_img, kp_reference, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def main():
    # Load and preprocess images
    target_image_gray = preprocess_image(TARGET_PATH, SCALE_FACTOR)
    reference_image_gray = preprocess_image(REFERENCE_PATH, SCALE_FACTOR)
    
    # Detect Harris corners
    kp_target = detect_harris_corners(target_image_gray)
    kp_reference = detect_harris_corners(reference_image_gray)
    
    # Feature matching using different methods
    methods = ['SIFT', 'ORB']
    match_results = {}
    homographies = {}
    
    for method in methods:
        try:
            matches = feature_matching(method, target_image_gray, reference_image_gray, kp_target, kp_reference)
            homography, mask = apply_ransac(matches, kp_target, kp_reference)
            match_results[method] = draw_matches(target_image_gray, reference_image_gray, kp_target, kp_reference, matches)
            homographies[method] = homography
            print(f"{method}: dx = {homography[0,2]}, dy = {homography[1,2]}, matches = {len(matches)}")
        except Exception as e:
            print(f"Error with {method}: {e}")
    
    # Display results
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))
    for i, (method, img) in enumerate(match_results.items()):
        axs[i].imshow(img)
        axs[i].set_title(method)
        axs[i].axis('off')
    plt.show()

if __name__ == '__main__':
    main()