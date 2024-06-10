import os
import time

import cv2
import numpy as np


def get_correction_data(h):
    """
    Decompose homography matrix into rotation angle, x and y shifts.
    """
    sx = np.sign(h[0, 0]) * np.sqrt(h[0, 0]**2 + h[1, 0]**2)
    sy = np.sign(h[1, 1]) * np.sqrt(h[0, 1]**2 + h[1, 1]**2)

    rotation_matrix = np.array([[h[0, 0] / sx, h[0, 1] / sy],
                                [h[1, 0] / sx, h[1, 1] / sy]])

    translation = h[:2, 2]

    angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi

    return translation[0], translation[1], angle

def get_aligned_image(img1, img2):
    """Receive img1 - reference image and 
    img2 - image to transform. Returns the img2 aligned according to img1

    Args:
        img1 numpy.ndarray: reference image
        img2 numpy.ndarray: image to be aligned

    Returns:
        array of:
        homography matrix,
        numpy.ndarray: img2 aligned according to img1 points
    """
    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Decompose the homography matrix
    shift_x, shift_y, rotation_angle = get_correction_data(h)
    print(f"Shift in X: {shift_x}")
    print(f"Shift in Y: {shift_y}")
    print(f"Rotation Angle: {rotation_angle}")

    # Warp the image
    try:
        height, width, _ = img1.shape
    except:
        height, width = img1.shape
        
    start = time.time()
    im2_aligned = cv2.warpPerspective(img2, h, (width, height))
    print(f'Elapsed time={time.time()-start}')
    return h, im2_aligned
