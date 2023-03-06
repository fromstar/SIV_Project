import cv2 as cv2

class Camera:
    def __init__(self,url):
        self.cam = cv2.VideoCapture(url)
    
    def update_cam(self, url):
        self.cam.release()
        self.cam = cv2.VideoCapture(url)

    def capture_frame(self):
        result, image = self.cam.read()
        return result,image

    def release_cam(self):
        self.cam.release()