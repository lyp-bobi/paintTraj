import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from itertools import product,combinations
TLOW=10
THIGH=20

color1='black'
color2='yellowgreen'

bDrawBBox=False


global SIGID
SIGID='A'

def frange(start, stop, step):
    return [start+i*step for i in range(int((stop-start)//step))]

def drawcube(ax,xlow,ylow,tlow,xhigh,yhigh,thigh):
    x=[xlow,xhigh]
    y=[ylow,yhigh]
    t=[tlow,thigh]
    for s,e in combinations(np.array(list(product(x,y,t))),2):
        if np.sum(np.abs(s-e))==xhigh-xlow or np.sum(np.abs(s-e))==yhigh-ylow or np.sum(np.abs(s-e))==thigh-tlow:
            ax.plot3D(*zip(s,e),color='blue')

def plotLine(ax,p1,p2):
    print(p1,p2)
    lines=[]

    def loc(t):
        if(t==p1[2]):
            return p1
        return [p1[0] + (p2[0] - p1[0]) / (p2[2] - p1[2]) * (t - p1[2]),
         p1[1] + (p2[1] - p1[1]) / (p2[2] - p1[2]) * (t - p1[2]), t]

    ts = min(TLOW, p1[2])
    te = min(TLOW, p2[2])
    if(ts < te):
        lines.append((loc(ts),loc(te)))
    ts = max(TLOW, p1[2])
    te = min(THIGH, p2[2])
    if (ts < te):
        lines.append((loc(ts), loc(te)))
    ts = max(THIGH, p1[2])
    te = max(THIGH, p2[2])
    if (ts < te):
        lines.append((loc(ts), loc(te)))
    if len(lines) ==0:
        lines.append((p1,p2))
    for s,e in lines:
        color=""
        if s[2]>=TLOW and s[2]<THIGH:
            color=color1
        else:
            color=color2
        _x=[s[0],e[0]]
        _y=[s[1],e[1]]
        _t=[s[2],e[2]]
        ax.plot3D(_x,_y,_t,c=color)
    if p1[2]>=TLOW and p1[2]<=THIGH:
        ax.scatter3D(p1[0],p1[1],p1[2],c=color1)
    else:
        ax.scatter3D(p1[0],p1[1],p1[2],c=color2)
    if p2[2] >= TLOW and p2[2] <= THIGH:
        ax.scatter3D(p2[0], p2[1], p2[2], c=color1)
    else:
        ax.scatter3D(p2[0], p2[1], p2[2], c=color2)


def randTraj(ax,tdelt, tperiod, ps, signal=100):
    points=[ps]
    p1=ps
    ts=ps[2]
    xlow=xhigh=ps[0]
    ylow=yhigh=ps[1]
    tlow=ps[2]
    t=0
    for t in frange(ts+tdelt,ts+tperiod+tdelt/2,tdelt):
        p2 = [p1[0]+(random.random()-0.5)*(t-p1[2])*2 ,p1[1]+(random.random()-0.5)*(t-p1[2])*2, t]
        p1=p2
        if(random.random()*100<signal):
            plotLine(ax, ps, p2)
            points.append(p2)
            ps=p2
            if p1[0] > xhigh:
                xhigh = p1[0]
            if p1[0] < xlow:
                xlow = p1[0]
            if p1[1] > yhigh:
                yhigh = p1[1]
            if p1[1] < ylow:
                ylow = p1[1]
    if ps[2]!= p2[2]:
        plotLine(ax, ps, p2)
        points.append(p2)
        ps = p2
        if p1[0]>xhigh:
            xhigh=p1[0]
        if p1[0]<xlow:
            xlow=p1[0]
        if p1[1]>yhigh:
            yhigh=p1[1]
        if p1[1]<ylow:
            ylow=p1[1]
    thigh=p2[2]
    if bDrawBBox:
        drawcube(ax,xlow,ylow,tlow,xhigh,yhigh,thigh)
    return p2

def mark(ax,p):
    global SIGID
    ax.text(p[0], p[1], p[2], SIGID, size=30)
    SIGID = chr(ord(SIGID) + 1)





fig = plt.figure(figsize=(7,4))
ax = plt.axes(projection='3d')



# color1='black'
# color2='yellowgreen'
# ax.set_xlabel('X axis',fontsize=20)
# ax.set_ylabel('Y axis',fontsize=20)
# ax.set_zlabel('T axis',fontsize=20)
#
# ax.set_xticklabels([])
# ax.set_yticklabels([])
#
# mark(ax,randTraj(ax,0.5,30,[0,0,0],90))
# mark(ax,randTraj(ax,1,12,[15,15,16],40))
# mark(ax,randTraj(ax,5,30,[-1,11,0],100))
# mark(ax,randTraj(ax,1,16,[12,12,0],40))
# mark(ax,randTraj(ax,1,30,[13,2,0],20))


color1='brown'
color2='yellowgreen'

TLOW=0
THIGH=100

bDrawBBox = True

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

p=randTraj(ax,2,16,[0,0,0],100)
p=randTraj(ax,5,20,p,100)
p=randTraj(ax,5,20,p,100)
ax.view_init(15,5)

# Make panes transparent
ax.xaxis.pane.fill = False # Left pane
ax.yaxis.pane.fill = False # Right pane

# Remove grid lines
ax.grid(False)

# Remove tick labels
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Transparent spines
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

# Transparent panes
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# No ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.show()
fig.savefig("./sbb.eps")