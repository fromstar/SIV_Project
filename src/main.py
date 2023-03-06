import cv2
import joblib

from camera import Camera
from motion_detection import detect_motion

from hog import pre_built_hog
from hog import compute_hog

from svm import train
from svm import validate

# Set True if want to train the classifier
tr = False
url = 'http://192.168.1.147:8080/video'

def main(tr):

    # Cameras Creation
    cam = Camera(url)

    if tr == True:
        train(False)
        validate(False)
    
    clf = joblib.load("models/model.pkl")
    cnt = 0
    cnt_0 = 0
    while(True):
        result, frame = cam.capture_frame()
        if result == True:
            prs, motion = detect_motion(frame)
            
            if len(prs) > 0:
                cv2.imshow("Movemend Detected", motion)
                for sec in prs:
                    hog = compute_hog(sec)
                    # hog = pre_built_hog(nframe)
                    predict = clf.predict([hog])
                    print(predict)
                    if(predict[0] == 1):
                        cv2.imwrite("frames/results/frame_" + str(cnt)+".png",frame) 
                        cv2.imwrite("frames/results/box_" + str(cnt)+".png",sec) 
                        cv2.imwrite("frames/results/frbx_" + str(cnt)+".png",motion) 
                        cnt +=1 
                    else:
                        cnt_0 += 1        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(cnt/cnt_0)
            break
    
    cv2.destroyAllWindows()

main(tr)
