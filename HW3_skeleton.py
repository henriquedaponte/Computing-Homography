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
    



def calculateOutputSize(original_shape, H):
    



def solveQ1(get_pts, n_pts):



def solveQ2(get_pts, n_pts):



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



