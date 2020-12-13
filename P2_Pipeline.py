import numpy as np
import cv2
import datetime
import AdvancedLaneFinding as alf
from moviepy.editor import VideoFileClip


# CALIBRATE CAMERA
cam_mtx, dist_coeffs = alf.calibrate_camera()

# CREATE LINE & LANE OBJECTS FOR USE IN PIPELINE
left_lane = alf.Line()
right_lane = alf.Line()
lane = alf.Lane()


# PIPELINE FOR ADVANCED LANE FINDING
def pipeline(img, mode='mark_lanes'):

    # 1) Apply distortion correction
    undistorted = alf.undistort(img, cam_mtx, dist_coeffs)

    # 2) Use color transforms, gradients, etc., to create a thresholded binary image.
    thresholded, colored_thresholded = alf.threshold(undistorted)

    # Mask region of interest
    masked = alf.mask_region_of_interest(thresholded)

    # 3) Apply perspective transform
    top_view, M, Minv = alf.perspective_tr(masked)

    # 4) Detect lane pixels and fit polynomial
    # If previous lane was detected, search next to curve, otherwise use sliding window method
    if (left_lane.detected is False) or (right_lane.detected is False):
        try:
            left_fit, right_fit, lanes_colored = alf.sliding_windows(top_view)
        # if nothing was found, use previous fit
        except TypeError:
            left_fit = left_lane.previous_fit
            right_fit = right_lane.previous_fit
            lanes_colored = np.zeros_like(img)
    else:
        try:
            left_fit, right_fit, lanes_colored = alf.search_around_poly(top_view, left_lane.previous_fit, right_lane.previous_fit)
        except TypeError:
            try:
                left_fit, right_fit, lanes_colored = alf.sliding_windows(top_view)
            # if nothing was found, use previous fit
            except TypeError:
                left_fit = left_lane.previous_fit
                right_fit = right_lane.previous_fit
                lanes_colored = np.zeros_like(img)

    left_lane.current_fit = left_fit
    right_lane.current_fit = right_fit

    # Calculate base position of lane lines to get lane distance
    left_lane.line_base_pos = left_fit[0] * (top_view.shape[0] - 1) ** 2 + left_fit[1] * (top_view.shape[0] - 1) + left_fit[2]
    right_lane.line_base_pos = right_fit[0] * (top_view.shape[0] - 1) ** 2 + right_fit[1] * (top_view.shape[0] - 1) + right_fit[2]
    left_lane.line_mid_pos = left_fit[0] * (top_view.shape[0] // 2) ** 2 + left_fit[1] * (top_view.shape[0] // 2) + left_fit[2]
    right_lane.line_mid_pos = right_fit[0] * (top_view.shape[0] // 2) ** 2 + right_fit[1] * (top_view.shape[0] // 2) + right_fit[2]

    # Calculate top and bottom position of lane lines for sanity check
    lane.top_width = right_fit[2] - left_fit[2]
    lane.bottom_width = right_lane.line_base_pos - left_lane.line_base_pos
    lane.middle_width = right_lane.line_mid_pos - left_lane.line_mid_pos

    # Check if values make sense
    if alf.sanity_check(left_lane, right_lane, lane) is False:
        # If fit is not good, use previous values and indicate that lanes were not found
        if len(left_lane.previous_fits) == 5:
            diff_left = [0.0, 0.0, 0.0]
            diff_right = [0.0, 0.0, 0.0]
            for i in range(0, 3):
                for j in range(0, 3):
                    diff_left[i] += left_lane.previous_fits[j][i] - left_lane.previous_fits[j+1][i]
                    diff_right[i] += right_lane.previous_fits[j][i] - right_lane.previous_fits[j+1][i]

                diff_left[i] /= 4
                diff_right[i] /= 4

            for i in range(0, 3):
                left_lane.current_fit[i] = left_lane.previous_fit[i] + diff_left[i]
                right_lane.current_fit[i] = right_lane.previous_fit[i] + diff_right[i]
            print("prev: ", left_lane.previous_fit)
            print("diff: ", diff_left)
            print("fit: ", left_lane.current_fit)


            left_lane.detected = False
            right_lane.detected = False
        else:
            left_lane.current_fit = left_lane.previous_fit
            right_lane.current_fit = right_lane.previous_fit
            left_lane.detected = False
            right_lane.detected = False

    else:
        # If fit is good, use current values and indicate that lanes were found
        if left_lane.detected == False or right_lane.detected == False:
            left_lane.previous_fits.clear()
            right_lane.previous_fits.clear()
        left_lane.detected = True
        right_lane.detected = True
        left_lane.initialized = True
        right_lane.initialized = True
        left_lane.frame_cnt += 1
        right_lane.frame_cnt += 1

    # Calculate the average of the recent fits and set this as the current fit
    left_lane.average_fit = alf.average_fits(top_view.shape, left_lane)
    right_lane.average_fit = alf.average_fits(top_view.shape, right_lane)

    lane.average_bottom_width, lane.average_top_width = alf.average_width(top_view.shape, lane)

    # 5) Determine lane curvature and position of the vehicle
    left_lane.radius_of_curvature = alf.measure_curvature_real(top_view.shape, left_fit)
    right_lane.radius_of_curvature = alf.measure_curvature_real(top_view.shape, right_fit)
    curvature = left_lane.radius_of_curvature + right_lane.radius_of_curvature / 2

    left_lane.line_base_pos = left_fit[0] * (top_view.shape[0] - 1) ** 2 + left_fit[1] * (top_view.shape[0] - 1) + left_fit[2]
    right_lane.line_base_pos = right_fit[0] * (top_view.shape[0] - 1) ** 2 + right_fit[1] * (top_view.shape[0] - 1) + right_fit[2]
    vehicle_position = alf.get_vehicle_position(top_view.shape[1], left_lane.line_base_pos, right_lane.line_base_pos)

    # 6) Output: warp lane boundaries back & display lane boundaries, curvature and position
    lanes_marked = alf.draw_lanes(top_view, undistorted, left_lane.average_fit, right_lane.average_fit, curvature,
                                  vehicle_position, Minv)

    # Set current values as previous values for next frame
    left_lane.previous_fit = left_lane.current_fit
    right_lane.previous_fit = right_lane.current_fit

    # Reset / empty current fit
    left_lane.current_fit = [np.array([False])]
    right_lane.current_fit = [np.array([False])]

    if mode is 'debug':
        white_masked = np.dstack((masked, masked, masked)) * 255
        debug_top = np.concatenate((lanes_marked[:, 0:1279:2, :], colored_thresholded[:, 0:1279:2, :]), axis=1)
        debug_bottom = np.concatenate((white_masked[:, 0:1279:2, :], lanes_colored[:, 0:1279:2, :]), axis=1)
        debug = np.concatenate((debug_top[0:719:2], debug_bottom[0:719:2]), axis=0)
        return debug

    if mode is 'mark_lanes':
        return lanes_marked


# FUNCTIONS FOR TESTING
def test_on_video(video, mode, length):

    if video is "image":
        print("Testing on images")
        process_images()

    elif video is "video1":
        print("Testing first project video")
        process_video(1, mode, length)

    elif video is "video2":
        print("Testing challenge video")
        process_video(2, mode, length)

    elif video is "video3":
        print("Testing extra hard challenge video. Good luck...")
        process_video(3, mode, length)


def process_images():
    # APPLY PIPELINE ON IMAGE
    for num in range(0, 5):
        test_image = cv2.imread('test_images/test' + str(num) + '.jpg')
        result = pipeline(test_image)
        # Save output images
        output_fname_image = 'output_images/test_output'
        cv2.imwrite(output_fname_image + str(num) + '.jpg', result)


def process_video(video, mode='mark_lanes', length="long"):
    # APPLY PIPELINE ON VIDEO

    # Select input
    if video is 1:
        filename = 'project_video'
    elif video is 2:
        filename = 'challenge_video'
    elif video is 3:
        filename = 'harder_challenge_video'

    # Make only short subclip:
    if length is "long":
        test_input = VideoFileClip(filename + '.mp4')
    elif length is "short":
        test_input = VideoFileClip(filename + '.mp4').subclip(0, 3)

    # Name ouput file
    date = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")
    output_fname_video = 'output_videos/output_' + filename + date +'.mp4'

    # Process input video, write to output file
    test_output = test_input.fl_image(lambda inp_img: pipeline(inp_img, mode))
    test_output.write_videofile(output_fname_video, audio=False)