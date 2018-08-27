import numpy as np
import cv2
class GradientFilter:
    def __init__(self, abs_ksize = 3, mag_ksize = 9, dir_ksize = 15, abs_thresh=(20,100),mag_thresh=(30,100),dir_thresh=(0.7,1.3), color_thresh=(170,255)):
        self.abs_ksize = abs_ksize
        self.mag_ksize = mag_ksize
        self.dir_ksize = dir_ksize
        self.abs_thresh = abs_thresh
        self.mag_thresh = mag_thresh
        self.dir_thresh = dir_thresh
        self.color_thresh = color_thresh
        
    def abs_sobel_threshold(self, img, orient='x', sobel_kernel = 3, thresh = (20, 100)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel/  np.max(abs_sobel))
        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary
    
    def mag_threshold(self, img, sobel_kernel=9, mag_thresh=(30, 100)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output
    
    def dir_threshold(self, img, sobel_kernel=15, thresh=(0.7, 1.3)):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output
    
    def color_threshold(self, img, thresh=(90, 255)): # (170, 255)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1        
        return s_binary
    
    def process(self, img):
        # Apply each of the thresholding functions
        gradx = self.abs_sobel_threshold(img, orient='x', sobel_kernel=self.abs_ksize, thresh=self.abs_thresh)
        grady = self.abs_sobel_threshold(img, orient='y', sobel_kernel=self.abs_ksize, thresh=self.abs_thresh)
        mag_binary = self.mag_threshold(img, sobel_kernel=self.mag_ksize, mag_thresh=self.mag_thresh)
        dir_binary = self.dir_threshold(img, sobel_kernel=self.dir_ksize, thresh=self.dir_thresh)
        color_binary = self.color_threshold(img, self.color_thresh)
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary==1)] = 1
        return combined