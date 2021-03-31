import cv2
import imutils

trackers = cv2.legacy.MultiTracker_create()

v= cv2.VideoCapture('dst.avi')

ret,frame = v.read()
k=4
print(frame.shape)


for i in range(k):
    cv2.imshow('Frame',frame)
    bbi= cv2.selectROI('Frame',frame)
    tracker_i = cv2.legacy.TrackerCSRT_create()
    trackers.add(tracker_i,frame,bbi)


while True:
    t1mark = 0
    ret, frame = v.read()
    if not ret:
        break
    (success,boxes)= trackers.update(frame)
    for box in boxes:
        (x,y,w,h) = [int(a) for a in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        x1 = w / 2
        y1 = h / 2
        cx = x + x1
        cy = y + y1

        cv2.line(frame, (30, 480), (30,0 ), (0, 255, 0), 2)
        if cx<100:
            t1mark = t1mark+ 1
    cv2.putText(frame, 'marks= ' + str(t1mark), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
    print(t1mark)

    cv2.imshow('Frame',frame)
    key= cv2.waitKey(5)
    if key == ord('q'):
        break



v.release()
cv2.destroyAllWindows()
