import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse


def get_points(img, n_pts):
    plt.imshow(img)
    pts = plt.ginput(n_pts)
    plt.close()
    return pts

def getHomography(source, target):

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

    # Solve for h
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Form the homography matrix
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1]
    ])

    return H

def calculateOutputSize(original_shape, H):

    # Find the new bounds of the warped image
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

def solveQ1(get_pts, n_pts):
    
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

def solveQ2(get_pts, n_pts):
    img1 = cv2.imread('KITP_face1.jpg')  
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    img2 = cv2.imread('KITP_face2.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    if get_pts:
        # Select corresponding points in both images
        source_pts = get_points(img1, n_pts)
        target_pts = get_points(img2, n_pts)

        # Compute the homography matrix from the points in the first image to the points in the second image
        H = getHomography(source_pts, target_pts)

        # Calculate output size and warp the first image to the perspective of the second image
        x_min, y_min, x_max, y_max = calculateOutputSize(img1.shape, H)
        dst_width = x_max - x_min
        dst_height = y_max - y_min
        dst = cv2.warpPerspective(img1, H, (dst_width, dst_height))

        # Display the result
        plt.imshow(dst)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_pts', action='store_true', help='If True get points else load from file')
    parser.add_argument('--Q1', action='store_true', help='Solve Q1')
    parser.add_argument('--Q2', action='store_true', help='Solve Q2')
    parser.add_argument('--n_points', type=int, default=4, help='Number of points per image')  
    opt = parser.parse_args()

    # Note that there's no error checking on the existance of the point files
    # If you try running without get_pts first, it will crash
    if opt.Q1:
        solveQ1(opt.get_pts, opt.n_points)
    if opt.Q2:
        solveQ2(opt.get_pts, opt.n_points)



