import os
import cv2 as cv
import time

def getdata():
    ff = open("test.txt", "w")
    if not os.path.isdir("./image"):
        os.mkdir("./image")
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 64), cap.set(cv.CAP_PROP_FRAME_HEIGHT, 48)
    i = 0
    while cap.isOpened():
        print("the number is {}".format((i) % 10+1))
        time.sleep(1)
        ret, frame = cap.read()
        print(ret)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow("frame", gray)
        if cv.waitKey(0)  == ord('c'):
            cv.imwrite('./image/{}.jpg'.format(i), gray)
            ff.writelines('./image/{}.jpg,{}\n'.format(i, (i) % 10+1))
            i = i + 1
        else:
            break
    ff.close()
    cap.release()
    cv.destroyAllWindows()
