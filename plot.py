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


def randTraj(ax, tdelt,ps, pe, signal=100):
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
        p2 = [p1[0] + (t-p1[2])*dx + 2*(random.random() - 0.5) * (t - p1[2])*0.5 , p1[1]+ (t-p1[2])*dy + 2*(random.random() - 0.5) * (t - p1[2])*0.5, t]
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



def plotTraj(ax, points):
    ps = points[0]
    p1 = ps
    ts = ps[2]
    xlow = xhigh = ps[0]
    ylow = yhigh = ps[1]
    tlow = ps[2]
    t = 0
    for i in range(len(points)-1):
        p2 = points[i+1]
        p1 = points[i]
        plotLine(ax, ps, p2)
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
ax.set_ylabel('Y', fontsize=15, labelpad=5)
ax.set_zlabel('T', fontsize=15, labelpad=5)

ax.set_xticklabels([])
ax.set_yticklabels([])

# mark(ax,randTraj(ax,0.5,30,[0,0,0],90))
# mark(ax,randTraj(ax,1,12,[15,15,16],40))
# mark(ax,randTraj(ax,5,30,[-1,11,0],100))
# mark(ax,randTraj(ax,1,16,[12,12,0],40))
# mark(ax,randTraj(ax,1,30,[13,2,0],20))

bDrawBBox = False
bDrawMBC = True
# t1 = randTraj(ax,0.5,[0,0,0],[-3,5,30])
t1 = plotTraj(ax,[[0, 0, 0], [-0.2645445691766996, 0.16683373432495685, 0.5], [-0.13438754305463896, 0.09796737169545477, 1.0], [-0.4311263280615094, 0.1692345644922953, 1.5], [-0.29740909995907666, 0.46773539307029643, 2.0], [-0.48418661245355005, 0.7581194330615355, 2.5], [-0.5752429060995023, 0.709906589804012, 3.0], [-0.78658852414742, 0.9619287254833393, 3.5], [-0.7039819660087292, 0.9961762197133536, 4.0], [-0.7131324787703536, 1.0853228839591715, 4.5], [-0.6830572602138152, 1.3045652851775869, 5.0], [-0.894448592745283, 1.4695141434022772, 5.5], [-0.7794404296217139, 1.4379488173246504, 6.0], [-0.99978449907054, 1.5784710662977255, 6.5], [-0.8377839033321866, 1.5836900864952257, 7.0], [-1.1215362337525208, 1.4488675746087853, 7.5], [-1.3711170710411211, 1.3703524676222862, 8.0], [-1.5850653633645102, 1.3815211715573645, 8.5], [-1.8395294735127998, 1.387718955806431, 9.0], [-1.6676108066779998, 1.6292352218565835, 9.5], [-1.7723503992378542, 1.6956460647971519, 10.0], [-1.8492348252480368, 1.991109893535149, 10.5], [-2.0286155830390826, 1.988680224528661, 11.0], [-1.9097156885300353, 2.1623124128983937, 11.5], [-2.026687712476704, 2.4650244254960945, 12.0], [-1.850429155085672, 2.7168803990632, 12.5], [-2.0188328650328704, 2.701062238657466, 13.0], [-2.0713533127836383, 2.5858035701537387, 13.5], [-2.284557408083726, 2.4206584773189155, 14.0], [-2.2571251864030506, 2.4828900855716998, 14.5], [-2.170987541569812, 2.3410681314012907, 15.0]])
t1 = plotTraj(ax, [[-2.170987541569812, 2.3410681314012907, 15.0], [-2.255996278101778, 2.1245854754135793, 15.5], [-2.0677251786419477, 1.721741945126053, 16.5], [-1.8554542753451864, 1.7008763338137267, 17.0], [-1.8582900076269542, 1.356952424098332, 18.5], [-1.5469669924644476, 1.4362097346838978, 19.0], [-1.5133127500944514, 1.4085124668991003, 20.0], [-1.433227281256637, 1.3699768081243666, 20.5], [-1.5403177384650046, 1.3105869462195137, 21.0], [-1.6489089472975558, 1.4360738987351975, 22.0], [-1.778152734657475, 1.1165220218026153, 22.5], [-1.8269290933203606, 0.8554728898038451, 23.5], [-1.5373032721207904, 0.5934873587533356, 24.5], [-1.4327582417977964, 0.4156941040380799, 25.5], [-1.5835000033958593, 0.12470669802929207, 26.0], [-1.66057922339054, -0.09451608439591362, 26.5], [-1.3954287398870762, -0.3531620215879806, 27.0], [-1.5653926546555479, -0.6296346826745205, 27.5], [-1.5622910456412429, -0.9042882445966972, 28.5], [-1.190145628238971, -1.3688210524916788, 29.5]])

bDrawBBox = False
bDrawMBC = False
# t1 = randTraj(ax,0.5,[-2.170987541569812, 2.3410681314012907, 15.0], [0,0,30],70)
t2 = plotTraj(ax,[[5, 8, 18], [4.041815956625906, 7.9414977329857255, 19], [3.918177575835078, 7.902110735805324, 20], [3.155948835567261, 8.0213413489812, 21], [2.6423977517113295, 8.736582408826608, 22], [1.6027192811573385, 8.688821185293692, 23], [1.1401217258111593, 8.755025742604936, 24], [0.15185749376730073, 9.24591512280972, 25], [-0.787954596582799, 9.892996891633347, 26]])
# t2 = plotTraj(ax,[[15, 15, 16], [14.464252251881495, 14.994301986482178, 17], [13.136523478981589, 14.150172780597742, 19], [12.513443031942101, 13.624464355415352, 20], [11.75659427902941, 14.424051249418103, 21], [11.532947913522694, 16.30567108045932, 24], [11.17599572590468, 16.36379649656207, 27]])
t3 = plotTraj(ax,[[-1, 11, 0], [1.8277790411493475, 12.137152441841703, 5], [4.875773130637415, 14.173123076714194, 10], [0.08449773031238994, 11.930014149857614, 15], [-4.070196942480728, 11.477849920224878, 20], [-5.868493186388405, 11.181302663523816, 25]])
t4 = plotTraj(ax,[[12, 12, 0], [11.856142753217835, 11.29660033165816, 1], [13.767473866025124, 10.743166606773105, 4], [14.746108657176912, 11.323808490965616, 5], [14.68121261066479, 11.656282466730298, 7], [14.784518763872114, 10.984890603488958, 8], [14.46088114261098, 9.496715508784204, 11], [14.380010813739633, 7.985245198197768, 14]])
# t5 = randTraj(ax,0.5,[13,2,0], [20,-3,30],70)
bDrawBBox = True
t5 = plotTraj(ax,[[13, 2, 0], [12.88827735199276, 1.774623117012097, 0.5], [13.010083548530858, 1.7363167672894986, 1.0], [13.414964576415937, 1.893292470344659, 2.5], [14.023299183141413, 1.8287979580272693, 3.5], [13.918257752609344, 1.9017891345660125, 4.0], [14.017668940039028, 2.002848603165758, 4.5], [14.207694838319311, 2.140111785705492, 5.0], [14.408309935615408, 1.9794361698565308, 6.0], [14.720559440769406, 1.8700078184968922, 7.0], [15.049633865012186, 1.6340255255286391, 7.5], [15.289052028723516, 1.4518704039415302, 8.0], [15.41794162659321, 1.5065039946754573, 8.5], [15.616542587522114, 1.3567671704287985, 9.0], [16.00204464028406, 1.4320604803595742, 10.0], [15.993078233188292, 1.4687110784758814, 10.5], [16.142240131035678, 1.2627008411238454, 11.0], [16.25636137569025, 1.398240732834044, 11.5], [16.147399430338638, 1.2165886512427928, 12.0], [16.461148791339905, 1.2171432761500938, 12.5], [16.396025187661273, 1.0720302808601643, 13.0], [16.684594272343332, 0.7881073795354048, 14.0], [16.921740791778785, 0.49437736504551555, 15.0], [16.861634918593722, 0.20595753685848578, 15.5], [17.094200241764256, -0.11954647125663366, 16.0], [17.006830500057486, -0.33021703827187876, 16.5], [17.044616229843943, -0.3583321020929244, 17.0], [17.188611653985888, -0.42951598658849643, 17.5], [17.235104889066164, -0.3941918334682931, 18.0], [17.537935675396977, -0.552222343524037, 19.0], [17.507083439344846, -0.601015165299595, 19.5], [17.789387925993935, -0.740763418784844, 20.0], [17.766138431798648, -0.6133414616914965, 20.5], [17.865397546933103, -0.6449512329433076, 21.0], [17.81256128987292, -0.4882755907692604, 21.5], [17.99974594543017, -0.3938750552364787, 22.0], [18.11136007749086, -0.2486720120815245, 22.5], [18.130126155496512, -0.36935826290033724, 23.0], [18.435705871738637, -0.3241130673260696, 23.5], [18.483874595661046, -0.3219011420214014, 24.5], [18.46418171984633, -0.41594015292478226, 25.0], [18.561580064850197, -0.39871077609093514, 25.5], [18.485635427211932, -0.5973670156989013, 26.0], [18.766236950537824, -0.789699088468409, 27.0], [19.06869479234561, -0.879436237868208, 27.5], [18.943672634591838, -1.1554929152609674, 28.0], [19.045877064545937, -1.2495447197430245, 29.0], [19.278500078506923, -1.3597045832229704, 29.5]])
bDrawBBox = False
mark(ax,t1)
mark(ax,t2)
mark(ax,t3)
mark(ax,t4)
mark(ax,t5)


color1 = 'brown'
color2 = 'yellowgreen'

TLOW = 0
THIGH = 100

bDrawBBox = True
bDrawMBC = True

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# p = randTraj(ax, 2, 16, [10, 10, 10], 100)
# p=randTraj(ax,5,20,p,100)
# p=randTraj(ax,5,20,p,100)
ax.view_init(15, 5)

ax.view_init(20, 20)

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

plt.show()
fig.savefig("./sbb.png")
