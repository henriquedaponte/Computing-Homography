import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Function to get user-selected points from an image
def get_points(img, n_pts):
    """
    Display an image and let the user click `n_pts` points.

    Parameters:
    - img: The image to display.
    - n_pts: The number of points to select.

    Returns:
    - List of selected points.
    """
    plt.imshow(img)
    pts = plt.ginput(n_pts)
    plt.close()
    return pts

# Function to compute the homography matrix between two sets of points
def getHomography(source, target):
    """
    Compute the homography matrix based on point correspondences.

    Parameters:
    - source: List of points from the source image.
    - target: List of corresponding points from the target image.

    Returns:
    - 3x3 homography matrix.
    """
    A = []
    b = []

    for i in range(len(source)):
        x, y = source[i]
        x_, y_ = target[i]
        
        A.append([x, y, 1, 0, 0, 0, -x*x_, -y*x_])
        A.append([0, 0, 0, x, y, 1, -x*y_, -y*y_])
        b.append(x_)
        b.append(y_)

    A = np.array(A)
    b = np.array(b)

    # Solve for h using the least squares method
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Form the homography matrix
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1]
    ])

    return H

# Function to compute the bounds of the warped image
def calculateOutputSize(original_shape, H):
    """
    Calculate the bounds of the image after applying the homography.

    Parameters:
    - original_shape: Shape of the original image.
    - H: Homography matrix.

    Returns:
    - List containing the min and max x and y coordinates.
    """
    corners = np.array([
        [0, 0],
        [original_shape[1]-1, 0],
        [original_shape[1]-1, original_shape[0]-1],
        [0, original_shape[0]-1]
    ], dtype=np.float64)

    new_corners = cv2.perspectiveTransform(corners.reshape(-1,1,2), H)

    x_min, y_min = np.int32(new_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(new_corners.max(axis=0).ravel())

    return [x_min, y_min, x_max, y_max]

# Function to solve Q1
def solveQ1(get_pts, n_pts):
    """
    Correct the perspective distortion of an image.

    Parameters:
    - get_pts: Boolean indicating if points should be selected manually.
    - n_pts: Number of points to select.
    """
    img = cv2.imread('KITP_face1.JPG')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if get_pts:
        source_pts = get_points(img, n_pts)
        target_pts = get_points(np.zeros_like(img), n_pts)

        H = getHomography(source_pts, target_pts)
        x_min, y_min, x_max, y_max = calculateOutputSize(img.shape, H)
        
        # Warp the image using the computed homography matrix
        dst_width = x_max - x_min
        dst_height = y_max - y_min
        dst = cv2.warpPerspective(img, H, (dst_width, dst_height))
        
        # Display the result
        plt.imshow(dst)
        plt.show()

# Function to solve Q2
def solveQ2(get_pts, n_pts):
    """
    Synthesize one view from another.

    Parameters:
    - get_pts: Boolean indicating if points should be selected manually.
    - n_pts: Number of points to select.
    """
    img1 = cv2.imread('Q21.jpeg')  
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    img2 = cv2.imread('Q22.jpeg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    if get_pts:
        source_pts = get_points(img1, n_pts)
        target_pts = get_points(img2, n_pts)

        H = getHomography(source_pts, target_pts)
        x_min, y_min, x_max, y_max = calculateOutputSize(img1.shape, H)
        dst_width = x_max - x_min
        dst_height = y_max - y_min
        dst = cv2.warpPerspective(img1, H, (dst_width, dst_height))

        plt.imshow(dst)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_pts', action='store_true', help='If True get points else load from file')
    parser.add_argument('--Q1', action='store_true', help='Solve Q1')
    parser.add_argument('--Q2', action='store_true', help='Solve Q2')
    parser.add_argument('--n_points', type=int, default=4, help='Number of points per image')  
    opt = parser.parse_args()

    if opt.Q1:
        solveQ1(opt.get_pts, opt.n_points)
    if opt.Q2:
        solveQ2(opt.get_pts, opt.n_points)
