import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
import xlwt

cap = cv2.VideoCapture('edmontonfight.mp4')

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))),isColor=True)
book = xlwt.Workbook(encoding="utf-8")

def resizeFrame(frame):
	'''resizes frame
	receives frame
	returns the frame with the new dimensions'''
	newFrameSize = (480,360)
	frame = cv2.resize(frame.copy(), newFrameSize)
	return frame

Nmframe=0

ret, old_frame = cap.read()
old_frame = resizeFrame(old_frame)
gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

index=0

(maxh,maxw) = gray.shape[:2]
n=input('The number of blocks:')
o1=np.linspace(1,360-1,n**0.5+1)  ###number of 'particles' vertically equal-spaced
o2=np.linspace(1,480-1,n**0.5+1)  ###number of 'particles' horizontally equal-spaced

print o1,o2
q0=[]
r0=[]

# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params) finding the corners

first = 'Y'
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        index=index+1
        if index == 5:
            frame = resizeFrame(frame)
            old_gray = gray
            index=0
            # for changing colors
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            Nmframe = Nmframe + 1

            flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #print flow[0]
            for column in range(int(n**0.5)):
                for row in range(int(n**0.5)):
                    sum = 0
                    t0=[]
                    for i in range(int(o1[column]),int(o1[column+1])):
                        line = flow[int(i)]
                        for j in line[int(round(o2[row])):int(round(o2[row+1]))]:
							t0.append(j[1])
                    average = np.mean(t0)
                    sd = np.std(t0)

                    if Nmframe == 1:
                        q0.append([(column+1),(row+1),average])
                        r0.append([(column+1),(row+1),sd])
                    else:
                        for q in q0:
                            if (q[0]==int(column+1) and q[1]==int(row+1)):
                                q.append(average)
                        for r in r0:
                            if (r[0]==int(column+1) and r[1]==int(row+1)):
                                r.append(sd)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:break

q0grid=np.array(q0, dtype=np.float32)
r0grid=np.array(r0, dtype=np.float32)

t=0
fig, ax = plt.subplots(int(n**0.5),int(n**0.5))
fig.subplots_adjust(hspace = 0.2, wspace = 0.2)
for q in q0:
	ax[int(q[0])-1,int(q[1])-1].scatter(range(Nmframe),q[2:],s=0.1)
	sheet1.write(int(t), 0, str(q))
	t = t+1
plt.show()

t=0
figg, bx = plt.subplots(int(n**0.5),int(n**0.5))
figg.subplots_adjust(hspace = 0.2, wspace = 0.2)
for r in r0:
	bx[int(r[0])-1,int(r[1])-1].scatter(range(Nmframe),r[2:],s=0.1)
	sheet2.write(int(t), 0, str(r))
	t = t+1
plt.show()

book.save("Extract-relative-y.xls")
print q0grid
print r0grid
print Nmframe

cap.release()
cv2.destroyAllWindows()
