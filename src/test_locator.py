import os

import cv2

from . import locator

def test_locator(): 
    BASE_DIR = os.path.dirname(os.getcwd())
    # Load images
    img_reference = cv2.imread(os.path.join(BASE_DIR, 'src', 'images', 'master0.png'))  # Base image
    img2 = cv2.imread(os.path.join(BASE_DIR, 'src', 'images', 'test-01.png'))  # Image to align
    h, img = locator.get_aligned_image(img_reference, img2)
    shift_x, shift_y, rotation_angle = locator.get_correction_data(h)
    assert (shift_x == -64.09588402596992 and shift_y == -263.8996949064434 and rotation_angle == -0.030238791346037887)