import cv2
import numpy as np


def threshold(inp_image, thresh1=(100, 255), thresh2=(25, 30), thresh3=(25, 30), thresh4=(25, 30), thresh5=(20, 100)):

    # Make a copy of the image
    img = np.copy(inp_image)

    # Get RGB channels
    r_channel = img[:, :, 2]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 0]

    # Convert to LUV color space
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    l_channel = luv[:, :, 0]

    # Sobel x
    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= thresh5[0]) & (scaled_sobel <= thresh5[1])] = 1

    # Threshold color channel with l channel
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= thresh1[0]) & (l_channel <= thresh1[1])] = 1

    # Get yellows
    r_binary = np.zeros_like(l_channel)
    g_binary = np.zeros_like(l_channel)
    b_binary = np.zeros_like(l_channel)
    r_binary[(r_channel >= thresh2[0]) & (r_channel <= thresh2[1])] = 1
    g_binary[(g_channel >= thresh3[0]) & (g_channel <= thresh3[1])] = 1
    b_binary[(b_channel >= thresh4[0]) & (b_channel <= thresh4[1])] = 1
    yellow_binary = np.zeros_like(l_channel)
    yellow_binary[(r_binary == 1) & (g_binary == 1) & (b_binary == 1)] = 1

    # White edges
    sobel_white_binary = np.zeros_like(l_channel)
    sobel_white_binary[(sx_binary == 1) & (l_binary == 1)] = 1

    # Yellow edges
    sobel_yellow_binary = np.zeros_like(l_channel)
    sobel_yellow_binary[(sx_binary == 1) & (yellow_binary == 1)] = 1

    # Stack channels (BGR)
    white_sobelx_and_color = np.dstack((sobel_white_binary, sobel_yellow_binary, np.zeros_like(sobel_white_binary))) * 255

    return white_sobelx_and_color


def nothing(x):
    pass


# Read sample image and create window
sample_img = cv2.imread('test_images/vlcsnap-2019-02-09-21h16m29s435.png')
cv2.namedWindow('Set threshold values')
cv2.namedWindow('Image')
height, width = sample_img.shape[:2]
sample_img = cv2.resize(sample_img, (width // 2, height // 2))

# create trackbars for color change
cv2.createTrackbar('L_min', 'Set threshold values', 185, 255, nothing)
cv2.createTrackbar('L_max', 'Set threshold values', 255, 255, nothing)
cv2.createTrackbar('R_min', 'Set threshold values', 140, 255, nothing)
cv2.createTrackbar('R_max', 'Set threshold values', 255, 255, nothing)
cv2.createTrackbar('G_min', 'Set threshold values', 140, 255, nothing)
cv2.createTrackbar('G_max', 'Set threshold values', 255, 255, nothing)
cv2.createTrackbar('B_min', 'Set threshold values', 0, 255, nothing)
cv2.createTrackbar('B_max', 'Set threshold values', 150, 255, nothing)
cv2.createTrackbar('sx_min', 'Set threshold values', 20, 255, nothing)
cv2.createTrackbar('sx_max', 'Set threshold values', 100, 255, nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'Set threshold values', 0, 1, nothing)

# Create a copy to make changes with trackbar
thresholded = sample_img

while 1:
    k = cv2.waitKey(1) & 0xFF
    if k == 27:                 # ESC character
        break

    # get current positions of four trackbars
    L_min = cv2.getTrackbarPos('L_min', 'Set threshold values')
    L_max = cv2.getTrackbarPos('L_max', 'Set threshold values')
    R_min = cv2.getTrackbarPos('R_min', 'Set threshold values')
    R_max = cv2.getTrackbarPos('R_max', 'Set threshold values')
    G_min = cv2.getTrackbarPos('G_min', 'Set threshold values')
    G_max = cv2.getTrackbarPos('G_max', 'Set threshold values')
    B_min = cv2.getTrackbarPos('B_min', 'Set threshold values')
    B_max = cv2.getTrackbarPos('B_max', 'Set threshold values')
    sx_min = cv2.getTrackbarPos('sx_min', 'Set threshold values')
    sx_max = cv2.getTrackbarPos('sx_max', 'Set threshold values')
    s = cv2.getTrackbarPos(switch, 'Set threshold values')

    if s == 0:
        cv2.imshow('Image', sample_img)
    else:
        thresholded = threshold(sample_img, thresh1=(L_min, L_max), thresh2=(R_min, R_max), thresh3=(G_min, G_max), thresh4=(B_min, B_max), thresh5=(sx_min, sx_max))
        cv2.imshow('Image', thresholded)

cv2.destroyAllWindows()
