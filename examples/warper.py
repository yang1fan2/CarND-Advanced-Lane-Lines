import cv2


class Warper:
    def __init__(self, img_size):
        self.img_size = img_size
        self.src = np.float32(
            [[(img_size[1] / 2) - 55, img_size[0] / 2 + 100],
            [((img_size[1] / 6) - 10), img_size[0]],
            [(img_size[1] * 5 / 6) + 60, img_size[0]],
            [(img_size[1] / 2 + 55), img_size[0] / 2 + 100]])
        self.dst = np.float32(
            [[(img_size[1] / 4), 0],
            [(img_size[1] / 4), img_size[0]],
            [(img_size[1] * 3 / 4), img_size[0]],
            [(img_size[1] * 3 / 4), 0]])
        
    def warp(self, img):
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        warped = cv2.warpPerspective(img, M, (self.img_size[1], self.img_size[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image
        return warped

    
    def unwarp(self, warped):
        Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        unwapred = cv2.warpPerspective(warped, Minv, (self.img_size[1], self.img_size[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image
        return unwapred