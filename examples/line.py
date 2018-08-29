import numpy as np
class Line:
    def __init__(self, width):
        self.width = width
        self.n_iters = 5
        # was the line detected in the last iteration?
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        self.ym_per_pix = 30.0/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/640 # meters per pixel in x dimension
        self.y_eval = None
        
    def append(self, fit, ploty):
        self.recent_xfitted.append(fit)
        self.current_fit = fit
        
        self.best_fit = np.array(self.recent_xfitted[-self.n_iters:]).mean(axis=0)
        self.ally = ploty
        self.allx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        
        if self.y_eval == None:
            self.y_eval = np.max(ploty)
        
        self.radius_of_curvature = ((1 + (2*self.best_fit[0]*self.y_eval*self.ym_per_pix + self.best_fit[1])**2)**1.5) / np.absolute(2*self.best_fit[0])
        

    
    def set_line_base_pos(self, base):
        self.line_base_pos = (self.width / 2.0 - base) * self.xm_per_pix
 