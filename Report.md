## Report
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_images/undistort_straight_lines1.jpg "Undistorted"
[image2]: ./output_images/gradient_filter/filter_straight_lines2.jpg "Gradient filter"
[image3]: ./output_images/warped/src_challenge0.jpg "Before warping"
[image4]: ./output_images/warped/dst_challenge0.jpg "After warping"
[image5]: ./output_images/window_search/straight_lines2.jpg "Window Search"
[image6]: ./output_images/final/challenge12.jpg "Final"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I use learned distortion-corrected matrix and get the following image:
![alt text][image1]
Other undistorted images can be found in [folder](https://github.com/yang1fan2/CarND-Advanced-Lane-Lines/tree/master/output_images/undistort_images)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I combined both L channel and S channel from HLS color space ([code](https://github.com/yang1fan2/CarND-Advanced-Lane-Lines/blob/master/examples/gradient_filter.py#L74)). For these two channels, I used three filters:
- Threshold on this channel directly
- Magnitude of the gradient (both x and y axis)
- Direction of the gradient
Here's an example of my output for this step. 

![alt text][image2]
Other binary images are located in this [folder](https://github.com/yang1fan2/CarND-Advanced-Lane-Lines/tree/master/output_images/gradient_filter)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is located [here](https://github.com/yang1fan2/CarND-Advanced-Lane-Lines/blob/master/examples/warper.py#L9).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
self.src = np.float32(
    [[(img_size[1] / 2) - 55, img_size[0] / 2 + 125],
    [((img_size[1] / 6) - 10), img_size[0]],
    [(img_size[1] * 5 / 6) + 60, img_size[0]],
    [(img_size[1] / 2 + 65), img_size[0] / 2 + 125]])
self.dst = np.float32(
    [[(img_size[1] / 4), 0],
    [(img_size[1] / 4), img_size[0]],
    [(img_size[1] * 3 / 4), img_size[0]],
    [(img_size[1] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 435      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 705, 435      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used both sliding window search ([code](https://github.com/yang1fan2/CarND-Advanced-Lane-Lines/blob/master/examples/window_searcher.py#L30)) and search from previous polynomial([code](https://github.com/yang1fan2/CarND-Advanced-Lane-Lines/blob/master/examples/window_searcher.py#L89)). For sliding window search, I first found two base points with the maximum histogram values. Then starting from bottom to top, collect pixels with a given margin. As for searching from prior, I searched the pixels near the previous lanes.

At last I fit my lane lines with a 2nd order polynomial with np.polyfit. The following image shows the detected lanes:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
I calculate the radius of curvature in [here](https://github.com/yang1fan2/CarND-Advanced-Lane-Lines/blob/master/examples/window_searcher.py#L150) and the position of the vehicle with respect to center [here](https://github.com/yang1fan2/CarND-Advanced-Lane-Lines/blob/master/examples/window_searcher.py#L167).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in [code](https://github.com/yang1fan2/CarND-Advanced-Lane-Lines/blob/master/examples/lane_finder.py#L42). Here is an example of my result on a test image: 

![alt text][image6]

---

### Pipeline (video)

Here's a [project_video](./output_videos/project_video.mp4), [challenge_video](./output_videos/challenge_video.mp4), [harder_challenge_video](./output_videos/harder_challenge_video.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- I spent a lot of time tuning the thresholds of the binary images. Firstly, I tried my best to reduce the noise. Otherwise, it will fail to fit the polynomial during the following steps. Secondly, in the challenge video, it's hard to only detect the yellow lanes without introducing the edges next to the yellow lanes. I solved the problem by using HLS color space.
- Searching from previous polynomial usually has better performance than sliding window search, especially if there are noises. It can directly ignore the noises by only focusing on a small area.
- The most important tricks is smoothing ([code](https://github.com/yang1fan2/CarND-Advanced-Lane-Lines/blob/master/examples/line.py#L33)) and sanity check ([code](https://github.com/yang1fan2/CarND-Advanced-Lane-Lines/blob/master/examples/main.py#L15)). I averaged the fitted parameters of the last five iterations. If one of the frames has bad detected lanes, smoothing can correct the error. Moreover, sanity check prevent adding bad frames into my smoothing array. I used two rules: the radius of curvature of left lane and right lane should be similar; The distance between left lane and right lane should be reasonable.
- The algorithm doesn't do well in harder_challenge_video. There are several reasons. In some frames, the sunlight is too strong, we cannot see anything. We should use different thresholds or use a more advanced filters. Secondly, the left yellow lane is pretty clear, while the right white lanes are covered by dust or leaves. In this case, we can only detect the left lane and infer the right lane. As last, the road has many turns. We should have a more robust area of interest, instead of hard-coding the area.
