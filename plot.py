import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from itertools import product, combinations

TLOW = 10
THIGH = 20

color1 = 'black'
color2 = 'yellowgreen'

bDrawBBox = False

global SIGID
SIGID = 'A'


def frange(start, stop, step):
    return [start + i * step for i in range(int((stop - start) // step))]


def drawcube(ax, xlow, ylow, tlow, xhigh, yhigh, thigh):
    x = [xlow, xhigh]
    y = [ylow, yhigh]
    t = [tlow, thigh]
    for s, e in combinations(np.array(list(product(x, y, t))), 2):
        if np.sum(np.abs(s - e)) == xhigh - xlow or np.sum(np.abs(s - e)) == yhigh - ylow or np.sum(
                np.abs(s - e)) == thigh - tlow:
            ax.plot3D(*zip(s, e), color='blue')


def plotLine(ax, p1, p2):
    lines = []

    def loc(t):
        if (t == p1[2]):
            return p1
        return [p1[0] + (p2[0] - p1[0]) / (p2[2] - p1[2]) * (t - p1[2]),
                p1[1] + (p2[1] - p1[1]) / (p2[2] - p1[2]) * (t - p1[2]), t]

    ts = min(TLOW, p1[2])
    te = min(TLOW, p2[2])
    if (ts < te):
        lines.append((loc(ts), loc(te)))
    ts = max(TLOW, p1[2])
    te = min(THIGH, p2[2])
    if (ts < te):
        lines.append((loc(ts), loc(te)))
    ts = max(THIGH, p1[2])
    te = max(THIGH, p2[2])
    if (ts < te):
        lines.append((loc(ts), loc(te)))
    if len(lines) == 0:
        lines.append((p1, p2))
    for s, e in lines:
        color = ""
        if s[2] >= TLOW and s[2] < THIGH:
            color = color1
        else:
            color = color2
        _x = [s[0], e[0]]
        _y = [s[1], e[1]]
        _t = [s[2], e[2]]
        ax.plot3D(_x, _y, _t, c=color)
    if p1[2] >= TLOW and p1[2] <= THIGH:
        ax.scatter3D(p1[0], p1[1], p1[2], c=color1)
    else:
        ax.scatter3D(p1[0], p1[1], p1[2], c=color2)
    if p2[2] >= TLOW and p2[2] <= THIGH:
        ax.scatter3D(p2[0], p2[1], p2[2], c=color1)
    else:
        ax.scatter3D(p2[0], p2[1], p2[2], c=color2)


# ax:plot
# tdelt sampling rate
# tperiod time period
# ps start point
def randTraj(ax, tdelt, tperiod, ps, signal=100):
    points = [ps]
    p1 = ps
    ts = ps[2]
    xlow = xhigh = ps[0]
    ylow = yhigh = ps[1]
    tlow = ps[2]
    t = 0
    for t in frange(ts + tdelt, ts + tperiod + tdelt / 2, tdelt):
        p2 = [p1[0] + (random.random() - 0.5) * (t - p1[2]) * 2, p1[1] + (random.random() - 0.5) * (t - p1[2]) * 2, t]
        p1 = p2
        if (random.random() * 100 < signal):
            plotLine(ax, ps, p2)
            points.append(p2)
            ps = p2
            if p1[0] > xhigh:
                xhigh = p1[0]
            if p1[0] < xlow:
                xlow = p1[0]
            if p1[1] > yhigh:
                yhigh = p1[1]
            if p1[1] < ylow:
                ylow = p1[1]
    if ps[2] != p2[2]:
        plotLine(ax, ps, p2)
        points.append(p2)
        ps = p2
        if p1[0] > xhigh:
            xhigh = p1[0]
        if p1[0] < xlow:
            xlow = p1[0]
        if p1[1] > yhigh:
            yhigh = p1[1]
        if p1[1] < ylow:
            ylow = p1[1]
    thigh = p2[2]
    print(points)
    if bDrawBBox:
        if bDrawMBC:
            plotmbc(ax, calcmbc(points))
        else:
            drawcube(ax, xlow, ylow, tlow, xhigh, yhigh, thigh)
    return p2


def randTraj(ax, tdelt,ps, pe, signal=100, rd=0.5):
    points = [ps]
    p1 = ps
    ts = ps[2]
    xlow = xhigh = ps[0]
    ylow = yhigh = ps[1]
    tlow = ps[2]
    dx = (pe[0]-ps[0])/ (pe[2]-ps[2])
    dy = (pe[1] - ps[1])/ (pe[2] - ps[2])
    t = 0
    for t in frange(ts + tdelt, pe[2], tdelt):
        p2 = [p1[0] + (t-p1[2])*dx + 2*(random.random() - 0.5) * (t - p1[2])*rd , p1[1]+ (t-p1[2])*dy + 2*(random.random() - 0.5) * (t - p1[2])*rd, t]
        p1 = p2
        if (random.random() * 100 < signal):
            plotLine(ax, ps, p2)
            points.append(p2)
            ps = p2
            if p1[0] > xhigh:
                xhigh = p1[0]
            if p1[0] < xlow:
                xlow = p1[0]
            if p1[1] > yhigh:
                yhigh = p1[1]
            if p1[1] < ylow:
                ylow = p1[1]
    if ps[2] != p2[2]:
        plotLine(ax, ps, p2)
        points.append(p2)
        ps = p2
        if p1[0] > xhigh:
            xhigh = p1[0]
        if p1[0] < xlow:
            xlow = p1[0]
        if p1[1] > yhigh:
            yhigh = p1[1]
        if p1[1] < ylow:
            ylow = p1[1]
    thigh = p2[2]
    print(points)
    if bDrawBBox:
        if bDrawMBC:
            plotmbc(ax, calcmbc(points))
        else:
            drawcube(ax, xlow, ylow, tlow, xhigh, yhigh, thigh)
    return p2



def plotTraj(ax, points, add= [0,0,0]):
    mod = []
    for p in points:
        mod.append(np.array(p)+np.array(add))
    points = mod
    ps = points[0]
    p1 = ps
    ts = ps[2]
    xlow = xhigh = ps[0]
    ylow = yhigh = ps[1]
    tlow = ps[2]
    t = 0
    for i in range(len(points)-1):
        p2 = np.array(points[i+1])
        p1 = np.array(points[i])
        plotLine(ax, ps, p2)
        ps = p2
        if p1[0] > xhigh:
            xhigh = p2[0]
        if p1[0] < xlow:
            xlow = p2[0]
        if p1[1] > yhigh:
            yhigh = p2[1]
        if p1[1] < ylow:
            ylow = p2[1]
    thigh = p2[2]
    if bDrawBBox:
        drawcube(ax, xlow, ylow, tlow, xhigh, yhigh, thigh)
    if bDrawMBC:
        plotmbc(ax, calcmbc(points))
    return p2

def mark(ax, p):
    global SIGID
    ax.text(p[0], p[1], p[2], SIGID, size=30)
    SIGID = chr(ord(SIGID) + 1)


import math


def calcmbc(pts: list):
    j = len(pts)
    ps = pts[0]
    pe = pts[j - 1]
    startx = ps[0]
    starty = ps[1]
    startt = ps[2]
    endx = pe[0]
    endy = pe[1]
    endt = pe[2]
    avx = (endx - startx) / (endt - startt)
    avy = (endy - starty) / (endt - startt)
    rd = 0
    rv = 0
    for i in range(1, j - 1):
        ptime = pts[i][2]
        if ptime > startt:
            vx = (pts[i][0] - startx) / (ptime - startt)
            vy = (pts[i][1] - starty) / (ptime - startt)
            prv = math.sqrt((vx - avx) * (vx - avx) + (vy - avy) * (vy - avy))
            if (prv > rv):
                rv = prv
        if ptime < endt:
            vx = (endx - pts[i][0]) / (endt - ptime)
            vy = (endy - pts[i][1]) / (endt - ptime)
            prv = math.sqrt((vx - avx) * (vx - avx) + (vy - avy) * (vy - avy));
            if (prv > rv):
                rv = prv
        posx = (ptime - startt) / (endt - startt) * endx + (endt - ptime) / (endt - startt) * startx
        posy = (ptime - startt) / (endt - startt) * endy + (endt - ptime) / (endt - startt) * starty
        prd = math.sqrt((posx - pts[i][0]) * (posx - pts[i][0]) + (posy - pts[i][1]) * (posy - pts[i][1]))
        if (prd > rd):
            rd = prd
        print((ps, pe, rd, rv))
    return (ps, pe, rd, rv)


def plotmbc(ax, mbc):
    ps = mbc[0]
    pe = mbc[1]
    rd = mbc[2]
    rv = mbc[3]
    dt = rd / rv

    # lower cone
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(ps[2], ps[2] + dt, 21)
    theta = np.linspace(0, 2 * np.pi, 21)
    v = [(pe[0] - ps[0]) / (pe[2] - ps[2]), (pe[1] - ps[1]) / (pe[2] - ps[2]), 1]
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    X, Y, Z = [ps[i] + v[i] * (t - ps[2]) + rv * (t - ps[2]) * np.cos(theta) * np.array([1, 0, 0])[i] + rv * (
                t - ps[2]) * np.sin(theta) * np.array([0, 1, 0])[i] for i in [0, 1, 2]]
    ax.plot_surface(X, Y, Z, color=(1.0, 1.0, 1.0, 0.3))

    # middle cylinder
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(ps[2] + dt, pe[2] - dt, 21)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    X, Y, Z = [ps[i] + v[i] * (t - ps[2]) + rd * np.cos(theta) * np.array([1, 0, 0])[i] + rd * np.sin(theta) *
               np.array([0, 1, 0])[i] for i in [0, 1, 2]]
    ax.plot_surface(X, Y, Z, color=(1.0, 1.0, 1.0, 0.3))

    # upper cone
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(pe[2] - dt, pe[2], 21)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    X, Y, Z = [ps[i] + v[i] * (t- ps[2]) + rv * (pe[2] - t) * np.cos(theta) * np.array([1, 0, 0])[i] + rv * (
                pe[2] - t) * np.sin(theta) * np.array([0, 1, 0])[i] for i in [0, 1, 2]]
    ax.plot_surface(X, Y, Z, color=(1.0, 1.0, 1.0, 0.3))


fig = plt.figure(figsize=(7, 4))
ax = plt.axes(projection='3d')

# color1='black'
# color2='yellowgreen'
ax.set_xlabel('X', fontsize=15, labelpad=-10)
ax.set_ylabel('Y', fontsize=15, labelpad=-10)
ax.set_zlabel('T', fontsize=15, labelpad=0)

ax.set_xticklabels([])
ax.set_yticklabels([])

# mark(ax,randTraj(ax,0.5,30,[0,0,0],90))
# mark(ax,randTraj(ax,1,12,[15,15,16],40))
# mark(ax,randTraj(ax,5,30,[-1,11,0],100))
# mark(ax,randTraj(ax,1,16,[12,12,0],40))
# mark(ax,randTraj(ax,1,30,[13,2,0],20))
# #
# bDrawBBox = False
# bDrawMBC = True
# # t1 = randTraj(ax,1,[0,5,0],[-3,-3,15])
# # t1 = randTraj(ax,1,[-3,-5,15],[0,5,30])
# t1 = plotTraj(ax, [[0, 5, 0], [-0.495923809996103, 4.1186494462891865, 1], [-0.2262449837971583, 3.2571868839119333, 2], [-0.19080486602353414, 2.6418246040475806, 3], [-0.04535753109000251, 1.7182305957713138, 4], [0.04804933952074181, 0.7055904270850426, 5], [-0.6249326521092305, 0.422950176372642, 6], [-0.7891935958320969, -0.02083920091856728, 7], [-1.179700214437103, -0.9879030746918613, 8], [-1.3975302139398815, -1.3735966661953092, 9], [-1.1791434217488805, -2.0480429512534144, 10], [-1.575239834317096, -2.424602946865162, 11], [-1.4225049647898396, -2.60384622649232, 12], [-1.6364409263406343, -2.7502517554318167, 13], [-1.4499719300613596, -2.8566955347204104, 14]])
# # t1 = randTraj(ax,1,[-1.4499719300613596, -2.8566955347204104, 14],[5,2,30])
# t1 = plotTraj(ax, [[-1.4499719300613596, -2.8566955347204104, 14], [-1.465270233594592, -2.2135421351868954, 15], [-0.9627445275917452, -2.193615440708404, 16], [-0.560136574942806, -2.27372480559289, 17], [0.3078993022140649, -1.6553652229520663, 18], [0.4572714978116794, -1.0191981733464903, 19], [0.5853899714007788, -0.8577867870668825, 20], [0.6066702378454175, -0.8120093946014009, 21], [1.462941346094592, -0.2787602804682241, 22], [1.6668336636821703, 0.15103511348194154, 23], [2.2803945033928192, 0.38833971590251193, 24], [3.104443632245161, 0.3142328726877499, 25], [3.3288051138006227, 1.0307569448511045, 26], [3.3080329525058714, 1.34984290048219, 27], [3.3348441764210897, 1.1633259324527474, 28], [3.586642778022414, 1.2009506622388892, 29]])
# bDrawBBox = False
# bDrawMBC = False
# # t1 = randTraj(ax,0.5,[-2.170987541569812, 2.3410681314012907, 15.0], [0,0,30],70)
# t2 = plotTraj(ax,[[5, 8, 18], [4.041815956625906, 7.9414977329857255, 19], [3.918177575835078, 7.902110735805324, 20], [3.155948835567261, 8.0213413489812, 21], [2.6423977517113295, 8.736582408826608, 22], [1.6027192811573385, 8.688821185293692, 23], [1.1401217258111593, 8.755025742604936, 24], [0.15185749376730073, 9.24591512280972, 25], [-0.787954596582799, 9.892996891633347, 26]])
# # t2 = plotTraj(ax,[[15, 15, 16], [14.464252251881495, 14.994301986482178, 17], [13.136523478981589, 14.150172780597742, 19], [12.513443031942101, 13.624464355415352, 20], [11.75659427902941, 14.424051249418103, 21], [11.532947913522694, 16.30567108045932, 24], [11.17599572590468, 16.36379649656207, 27]])
# t3 = plotTraj(ax,[[-1, 11, 0], [1.8277790411493475, 12.137152441841703, 5], [4.875773130637415, 14.173123076714194, 10], [0.08449773031238994, 11.930014149857614, 15], [-4.070196942480728, 11.477849920224878, 20], [-5.868493186388405, 11.181302663523816, 25]])
# # t4 = randTraj(ax, 2,[12,12,0], [14,14,14])
# t4 = plotTraj(ax,[[12, 12, 0], [13.077201802245886, 12.068595795996778, 2], [14.161495102019039, 11.494990538273377, 4], [14.580574926087209, 11.077641597644053, 6], [15.104733846335924, 11.960115801964491, 8], [16.214838072236088, 11.965208031929727, 10], [15.602605981058922, 11.723793295455833, 12]])
# # t5 = randTraj(ax,1,[3,2,0], [20,-3,30],80,rd=0.8)
# bDrawBBox = True
# t5 = plotTraj(ax,[[3, 2, 0], [3.218652111050188, 1.073440258211938, 1], [4.011459685038829, 0.47396333908400246, 2], [3.9672937667948514, 0.3900607556092325, 4], [5.2368765213713395, -0.09575505531230713, 5], [5.558960436393905, -0.8745415572237596, 6], [5.7594294621150315, -1.5134675783569562, 8], [6.115234639544745, -2.2916598476902292, 10], [7.678454636876871, -2.494150298657032, 12], [9.473928503214218, -4.055375282252717, 14], [10.818876222855579, -3.6530049443676416, 15], [11.880240947984982, -3.7611900811906462, 16], [12.821225648601898, -3.5518051807922753, 18], [13.082006662945853, -3.764432060630071, 19], [13.728214704194585, -3.6749133406225014, 20], [14.39140297223221, -4.300537893520304, 21], [14.445083617877744, -4.087945247285194, 23], [14.653958947282458, -4.548710177111478, 24], [14.929939745479128, -5.315755634817586, 26], [14.892001957367142, -6.041181876080585, 27], [15.98729399924144, -5.655752999088336, 28], [15.874880929970015, -5.559120463349535, 29]])
# bDrawBBox = False
# t6 = plotTraj(ax,[[3,2,0],[15.874880929970015, -5.559120463349535, 29]])
# mark(ax,t1)
# mark(ax,t2)
# mark(ax,t3)
# mark(ax,t4)
# mark(ax,[19,-3,31])
# mark(ax,[13,2,0])
#
#
# color1 = 'brown'
# color2 = 'yellowgreen'
#
# TLOW = 0
# THIGH = 100

bDrawBBox = True
bDrawMBC = True

ps = [[116.511720,39.921230,7524.000000],[116.511350,39.938830,8124.000000],[116.516270,39.910340,8724.000000],[116.471860,39.912480,9324.000000],[116.472170,39.924980,9924.000000],[116.471790,39.907180,10524.000000],[116.456170,39.905310,11124.000000],[116.471910,39.905770,12580.000000],[116.506610,39.914500,13180.000000],[116.496250,39.914600,25190.000000],[116.509620,39.910710,25789.000000],[116.522310,39.915880,26389.000000],[116.564440,39.914450,26989.000000]];


plotTraj(ax, ps)

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# p = randTraj(ax, 2, 16, [10, 10, 10], 100)
# p=randTraj(ax,5,20,p,100)
# p=randTraj(ax,5,20,p,100)
ax.view_init(15, 5)

ax.view_init(90, 0)

# # Make panes transparent
# ax.xaxis.pane.fill = False # Left pane
# ax.yaxis.pane.fill = False # Right pane

# # Remove grid lines
# ax.grid(False)
#
# Remove tick labels
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_zticklabels([])
#
# # Transparent spines
# ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

# # Transparent panes
# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# #
# # No ticks
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# # move axies
# ax.xaxis._axinfo['juggled'] = (0, 0, 0)
# ax.yaxis._axinfo['juggled'] = (1, 1, 1)
# ax.zaxis._axinfo['juggled'] = (2, 2, 0)
plt.margins(0)
plt.show()
fig.savefig("./irreg.pdf",transparent=True, bbox_inches='tight', pad_inches=0)
