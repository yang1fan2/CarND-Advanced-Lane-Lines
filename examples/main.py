from moviepy.editor import VideoFileClip
import pickle
from lane_finder import LaneFinder
from line import Line
import numpy as np

gray_shape, objpoints, imgpoints = pickle.load(open('camera.p', 'rb'))
lane_finder = LaneFinder(gray_shape, objpoints, imgpoints)
left_line = Line(gray_shape[1])
right_line = Line(gray_shape[1])
is_previous_lane_detected = False
global n_skipped_frames
n_skipped_frames = 0

def sanity_check(lane_finder):
    curs = lane_finder.window_searcher.measure_curvature_real()
    if abs(curs[0] - curs[1])> 4000:
        print(curs)
        return False
    errors = (lane_finder.window_searcher.right_fitx - lane_finder.window_searcher.left_fitx).mean()
    if errors > 1000 or errors < 200:
        print(errors)
        return False
    return True

def process_image(image):
    global n_skipped_frames
    if n_skipped_frames <= 5:
        lane_finder.run_pipeline(image, None,None)#
    else:
        lane_finder.run_pipeline(image)            
    if sanity_check(lane_finder) == True or left_line.best_fit == None:
        left_line.append(lane_finder.window_searcher.left_fit, lane_finder.window_searcher.ploty)
        right_line.append(lane_finder.window_searcher.right_fit, lane_finder.window_searcher.ploty)
        base = (left_line.allx[-1] + right_line.allx[-1]) / 2.0
        left_line.set_line_base_pos(base)
        right_line.set_line_base_pos(base)
        n_skipped_frames = 0
    else:
        n_skipped_frames += 1
    result = lane_finder.add_detect_lanes(image, left_line, right_line)
    return result
if __name__ == '__main__':
    
    
    clip1 = VideoFileClip("../harder_challenge_video.mp4")#.subclip(2, 5)
    white_clip = clip1.fl_image(process_image) 
    white_clip.write_videofile('../output_videos/harder_challenge_video.mp4', audio=False)
