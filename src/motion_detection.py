import cv2

fgbg = cv2.createBackgroundSubtractorMOG2(history = 2,detectShadows=False)

def detect_motion(frame):
    fb = cv2.GaussianBlur(frame,(5,5),0)
    fgmask = fgbg.apply(fb)
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    fgmask = cv2.erode(fgmask, None, iterations = 1)
    fgmask = cv2.dilate(fgmask, None, iterations = 2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frameCopy = frame.copy()
    
    fs = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 7000:
            x, y, width, height = cv2.boundingRect(cnt)
            cv2.rectangle(frameCopy, (x , y), (x + width, y + height),(0, 0, 255), 2)
            fs = fs + [frame[y:y+height, x:x+width]]

    return fs, frameCopy