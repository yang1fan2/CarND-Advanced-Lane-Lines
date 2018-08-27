import cv2
from camera_undistort import CameraUndistort
from gradient_filter import GradientFilter
from warper import Warper
import numpy as np
from window_searcher import WindowSearcher

class LaneFinder:
    def __init__(self, gray_shape, objpoints, imgpoints):
        self.camera_undistort = CameraUndistort(gray_shape, objpoints, imgpoints)
        self.gradient_filter = GradientFilter()
        self.warper = Warper(gray_shape)
        self.window_searcher = WindowSearcher()
        self.cnt = 0
        
    def run_pipeline(self, img, left_fit =None, right_fit=None):
        self.cnt+=1
        dst = self.camera_undistort.undistort(img)
        #cv2.imwrite('../output_images/undistort_images/undistort_' + file_name, dst)
        self.binary = self.gradient_filter.process(dst)
    #    cv2.imwrite('../output_images/gradient_filter/filter_' + file_name, binary.astype('uint8') * 255)

        cv2.imwrite('../output_images/debug/binary_%d.png' % self.cnt, self.binary.astype('uint8') * 255)
        #cv2.polylines(img, [np.int32(warper.src.reshape(-1,1,2))], True,(0,0,255), 3)
        #cv2.imwrite('../output_images/warped/src_' + file_name, img)
    #     warped = warper.warp(img)
    #     cv2.polylines(warped, [np.int32(warper.dst.reshape(-1,1,2))], True, (0,0,255), 3)
    #     cv2.imwrite('../output_images/warped/dst_' + file_name, warped)

        self.warped = self.warper.warp(self.binary)
        cv2.imwrite('../output_images/debug/warped_%d.png' % self.cnt, self.warped.astype('uint8') * 255)        
    #    cv2.imwrite('../output_images/warped/dst_binary_' + file_name, warped.astype('uint8') * 255)
        out_img = self.window_searcher.fit_polynomial(self.warped, left_fit, right_fit)
        cv2.imwrite('../output_images/debug/out_%d.png' % self.cnt, out_img)
    #    cv2.imwrite('../output_images/window_search/' + file_name, out_img)
    #    unwarped = warper.unwarp(out_img)
    #    cv2.imwrite('../output_images/unwarped/' + file_name, unwarped)
    #     print(window_searcher.measure_curvature_real())
    #     print(window_searcher.measure_center())


    def add_detect_lanes(self, img, left_line, right_line):
        ploty = left_line.ally
        left_fitx  = left_line.allx
        right_fitx = right_line.allx
        dst = self.camera_undistort.undistort(img)
        warp_zero = np.zeros_like(self.warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.warper.unwarp(color_warp)
        # Combine the result with the original image
        result = cv2.addWeighted(dst, 1, newwarp, 0.3, 0)
#        cv2.imwrite('../output_images/final/' + file_name, result)
        font = cv2.FONT_HERSHEY_SIMPLEX
        result = cv2.putText(result,'radius of curvature = %d m' % ((left_line.radius_of_curvature + right_line.radius_of_curvature)/2.0),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
        result = cv2.putText(result,'distance from the center = %.2f m' %  left_line.line_base_pos,(50,100), font, 1,(255,255,255),2,cv2.LINE_AA)
        
        return result