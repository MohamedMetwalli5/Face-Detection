import cv2 as cv
import imutils

capture = cv.VideoCapture(0)

while True:
    isTrue,frame = capture.read()
    frame = imutils.resize(frame, width=920, height=180)
    frame = cv.flip(frame, 1)

    faceCascade= cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    imageGray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imageGray,1.3,5)
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv.imshow("Video",frame)
    if cv.waitKey(6) & 0xFF == ord("q"):
        break

capture.release()
cv.destroyAllWindows()