import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
import xlwt
from Tkinter import *

cap = cv2.VideoCapture('edmontonfight_nonfighting.mp4')

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))),isColor=True)
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("x1")
sheet2 = book.add_sheet("y1")
sheet3 = book.add_sheet("x2")
sheet4 = book.add_sheet("y2")
sheet5 = book.add_sheet("x3")
sheet6 = book.add_sheet("y3")
sheet7 = book.add_sheet("x4")
sheet8 = book.add_sheet("y4")
sheet9 = book.add_sheet("x5")
sheet10 = book.add_sheet("y5")
sheet11 = book.add_sheet("x6")
sheet12 = book.add_sheet("y6")
sheet13 = book.add_sheet("x7")
sheet14 = book.add_sheet("y7")
sheet15 = book.add_sheet("x8")
sheet16 = book.add_sheet("y8")

def resizeFrame(frame):
	'''resizes frame
	receives frame
	returns the frame with the new dimensions'''
	newFrameSize = (480,360)
	frame = cv2.resize(frame.copy(), newFrameSize)
	return frame

Nmframe=0

ret, old_frame = cap.read()
#old_frame = resizeFrame(old_frame)
gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

index=0

(maxh,maxw) = gray.shape[:2]

q0=[]
r0=[]

start = 0
end = 427
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params) finding the corners

first = 'Y'
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        index=index+1
        if index == 5:
            #frame = resizeFrame(frame)
            old_gray = gray
            index=0
            # for changing colors
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            Nmframe = Nmframe + 1

            flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #print flow[0]

            q0=[]
            r0=[]
            for l in range(1200,1210):
				line = flow[l]
				for k in range(start,end):
					j=line[k]
					q0.append(round(abs(j[0]),3))
					r0.append(round(abs(j[1]),3))
            sheet1.write(int(Nmframe)-1, 0, str(q0))
            sheet2.write(int(Nmframe)-1, 0, str(r0))
            q0=[]
            r0=[]
            for l in range(1210,1220):
				line = flow[l]
				for k in range(start,end):
					j=line[k]
					q0.append(round(abs(j[0]),3))
					r0.append(round(abs(j[1]),3))
            sheet3.write(int(Nmframe)-1, 0, str(q0))
            sheet4.write(int(Nmframe)-1, 0, str(r0))
            q0=[]
            r0=[]
            for l in range(1220,1230):
				line = flow[l]
				for k in range(start,end):
					j=line[k]
					q0.append(round(abs(j[0]),3))
					r0.append(round(abs(j[1]),3))
            sheet5.write(int(Nmframe)-1, 0, str(q0))
            sheet6.write(int(Nmframe)-1, 0, str(r0))
            q0=[]
            r0=[]
            for l in range(1230,1240):
				line = flow[l]
				for k in range(start,end):
					j=line[k]
					q0.append(round(abs(j[0]),3))
					r0.append(round(abs(j[1]),3))
            sheet7.write(int(Nmframe)-1, 0, str(q0))
            sheet8.write(int(Nmframe)-1, 0, str(r0))
            q0=[]
            r0=[]
            for l in range(1240,1250):
				line = flow[l]
				for k in range(start,end):
					j=line[k]
					q0.append(round(abs(j[0]),3))
					r0.append(round(abs(j[1]),3))
            sheet9.write(int(Nmframe)-1, 0, str(q0))
            sheet10.write(int(Nmframe)-1, 0, str(r0))
            q0=[]
            r0=[]
            for l in range(1250,1260):
				line = flow[l]
				for k in range(start,end):
					j=line[k]
					q0.append(round(abs(j[0]),3))
					r0.append(round(abs(j[1]),3))
            sheet11.write(int(Nmframe)-1, 0, str(q0))
            sheet12.write(int(Nmframe)-1, 0, str(r0))
            q0=[]
            r0=[]
            for l in range(1260,1270):
				line = flow[l]
				for k in range(start,end):
					j=line[k]
					q0.append(round(abs(j[0]),3))
					r0.append(round(abs(j[1]),3))
            sheet13.write(int(Nmframe)-1, 0, str(q0))
            sheet14.write(int(Nmframe)-1, 0, str(r0))
            q0=[]
            r0=[]
            for l in range(1270,1280):
				line = flow[l]
				for k in range(start,end):
					j=line[k]
					q0.append(round(abs(j[0]),3))
					r0.append(round(abs(j[1]),3))
            sheet15.write(int(Nmframe)-1, 0, str(q0))
            sheet16.write(int(Nmframe)-1, 0, str(r0))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:break

q0grid=np.array(q0, dtype=np.float32)
r0grid=np.array(r0, dtype=np.float32)

book.save("Extract-pixel-16-1.xls")
print q0grid
print r0grid
print Nmframe

cap.release()
cv2.destroyAllWindows()
