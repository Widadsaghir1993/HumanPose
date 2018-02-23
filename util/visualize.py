import math

import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _npcircle(image, cx, cy, radius, color, transparency=0.0):
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    image[cy-radius:cy+radius, cx-radius:cx+radius][index] = (
        image[cy-radius:cy+radius, cx-radius:cx+radius][index].astype('float32') * transparency +
        np.array(color).astype('float32') * (1.0 - transparency)).astype('uint8')


def check_point(cur_x, cur_y, minx, miny, maxx, maxy):
    return minx < cur_x < maxx and miny < cur_y < maxy

def visualize_joints(image, pose):
    marker_size = 8
    minx = 2 * marker_size
    miny = 2 * marker_size
    maxx = image.shape[1] - 2 * marker_size
    maxy = image.shape[0] - 2 * marker_size
    num_joints = pose.shape[0]
    #print(num_joints)
    visim = image.copy()
    p1_x=0;p1_y=0
    p2_x = 0;p2_y = 0
    p3_x = 0;p3_y = 0
    p4_x = 0;p4_y = 0
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
              [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
              [0, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [0, 255, 0], [0,  255,0]]
    for p_idx in range(num_joints):
        cur_x = pose[p_idx, 0]
        cur_y = pose[p_idx, 1]
        if(p_idx==2):
            p1_x=cur_x;p1_y=cur_y
        elif(p_idx==3):
            p2_x = cur_x
            p2_y = cur_y
        elif (p_idx == 8):
            p3_x = cur_x
            p3_y = cur_y
        elif (p_idx == 9):
            p4_x = cur_x
            p4_y = cur_y
        if check_point(cur_x, cur_y, minx, miny, maxx, maxy):
            _npcircle(visim,
                      cur_x, cur_y,
                      marker_size,
                      colors[p_idx],
                      0.0)
    my=max(p1_y,p2_y)
    My = max(p3_y, p4_y)
    meanx1=(int)((p1_x+p2_x)/2);meanx2=(int)((p4_x+p3_x)/2)
    mean1x=(int)((meanx1*3+meanx2)/4)
    mean2x = (int)((meanx1  + meanx2*3) / 4)
    dx1=abs(p2_x-p1_x);dx2=abs(p4_x-p3_x)
    Dx=min(abs(p1_y-p3_y),abs(p2_y-p4_y))
    cur_x=mean1x-(int)(dx1*1.1/2 * 0.8)
    cur_y=my-(int)(Dx*0.4)
    if check_point(cur_x, cur_y, minx, miny, maxx, maxy):
        _npcircle(visim,
                  cur_x, cur_y,
                  marker_size,
                  colors[14],
                  0.0)
    cur_x1 = mean1x + (int)(dx1 * 1.1 / 2 * 0.8 )
    cur_y1 = my - (int)(Dx * 0.4)
    if check_point(cur_x1, cur_y1, minx, miny, maxx, maxy):
        _npcircle(visim,
                  cur_x1, cur_y1,
                  marker_size,
                  colors[15],
                  0.0)
    waist = np.sqrt((cur_x - cur_x1) * (cur_x - cur_x1) + (cur_y - cur_y1) * (cur_y - cur_y1))
    #print("waist is ", waist)
    cur_x = mean2x - (int)(dx2  / 2 * 0.75)
    cur_y = My + (int)(Dx * 0.25)
    if check_point(cur_x, cur_y, minx, miny, maxx, maxy):
        _npcircle(visim,
                  cur_x, cur_y,
                  marker_size,
                  colors[16],
                  0.0)
    cur_x1 = mean2x + (int)(dx2/ 2 * 0.75)
    cur_y1 = My + (int)(Dx * 0.25)
    if check_point(cur_x1, cur_y1, minx, miny, maxx, maxy):
        _npcircle(visim,
                  cur_x1, cur_y1,
                  marker_size,
                  colors[17],
                  0.0)
    chest = np.sqrt((cur_x-cur_x1)*(cur_x-cur_x1)+(cur_y-cur_y1)*(cur_y-cur_y1))
    #print("chest is ", chest )
    measure=[]
    leg11=np.sqrt((pose[0, 0]-pose[1,0])*(pose[0, 0]-pose[1,0])+(pose[0, 1]-pose[1,1])*(pose[0, 1]-pose[1,1]))
    leg12 = np.sqrt(
        (pose[2, 0] - pose[1, 0]) * (pose[2, 0] - pose[1, 0]) + (pose[2, 1] - pose[1, 1]) * (pose[2, 1] - pose[1, 1]))
    leg1 = leg11 + leg12
    #print("Right leg's length is ",leg1)
    leg21 = np.sqrt(
        (pose[3, 0] - pose[4, 0]) * (pose[3, 0] - pose[4, 0]) + (pose[3, 1] - pose[4, 1]) * (pose[3, 1] - pose[4, 1]))
    leg22 = np.sqrt(
        (pose[5, 0] - pose[4, 0]) * (pose[5, 0] - pose[4, 0]) + (pose[5, 1] - pose[4, 1]) * (pose[5, 1] - pose[4, 1]))
    leg2 = leg21 + leg22
    #print("Left leg's length is ", leg2)
    dl=(abs(pose[2,1]-pose[13,1])+abs(pose[3,1]-pose[13,1]))/2
    dl1 = (abs(pose[2, 1] - pose[12, 1]) + abs(pose[3, 1] - pose[12, 1])) / 2
    #print("Right Body length is ", leg1+dl)
    #print("LEFT Body length is ", leg2 + dl)
    #print("Right shoulder length is ", leg1 + dl1)
    #print("LEFT shoulder length is ", leg2 + dl1)
    Len1 =max(leg1,leg2) + dl1
    Len =max(leg1,leg2) + dl
    leg =max(leg1,leg2)

    arm11 = np.sqrt(
        (pose[6, 0] - pose[7, 0]) * (pose[6, 0] - pose[7, 0]) + (pose[6, 1] - pose[7, 1]) * (pose[6, 1] - pose[7, 1]))
    arm12 = np.sqrt(
        (pose[8, 0] - pose[7, 0]) * (pose[8, 0] - pose[7, 0]) + (pose[8, 1] - pose[7, 1]) * (pose[8, 1] - pose[7, 1]))
    arm1 = arm11 + arm12
    arm21 = np.sqrt(
        (pose[9, 0] - pose[10, 0]) * (pose[9, 0] - pose[10, 0]) + (pose[9, 1] - pose[10, 1]) * (pose[9, 1] - pose[10, 1]))
    arm22 =  np.sqrt(
        (pose[11, 0] - pose[10, 0]) * (pose[11, 0] - pose[10, 0]) + (pose[11, 1] - pose[10, 1]) * (pose[11, 1] - pose[10, 1]))
    arm2 = arm21 + arm22
    arm =max(arm1,arm2)
    #print("Right ARM length is ", arm1)
    #print("LEFT ARM length is ", arm2)
    shoulder= np.sqrt(
        (pose[8, 0] - pose[9, 0]) * (pose[8, 0] - pose[9, 0]) + (pose[8, 1] - pose[9, 1]) * (pose[8, 1] - pose[9, 1]))
    M = np.sqrt(
        (pose[2, 0] - pose[3, 0]) * (pose[2, 0] - pose[3, 0]) + (pose[2, 1] - pose[3, 1]) * (
        pose[2, 1] - pose[3, 1]))
    #print("shoulder width is ", shoulder)
    #print("Hip width is ", M )
    measure.append(Len)
    measure.append(Len1)
    measure.append(leg)
    measure.append(arm)
    measure.append(shoulder)
    measure.append(chest)
    measure.append(waist)
    measure.append(M)
    #print(measure)
    return visim,measure


def show_heatmaps(img, pose,sex,bf_flag):
    img, measure=visualize_joints(img, pose)
    plt.imshow(img)
    plt.show()

def show_arrows(cfg, img, pose, arrows):
    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    plt.imshow(img)
    a.set_title('Initial Image')


    b = fig.add_subplot(2, 2, 2)
    plt.imshow(img)
    b.set_title('Predicted Pairwise Differences')

    color_opt=['r', 'g', 'b', 'c', 'm', 'y', 'k']
    joint_pairs = [(6, 5), (6, 11), (6, 8), (6, 15), (6, 0)]
    color_legends = []
    for id, joint_pair in enumerate(joint_pairs):
        end_joint_side = ("r " if joint_pair[1] % 2 == 0 else "l ") if joint_pair[1] != 0 else ""
        end_joint_name = end_joint_side + cfg.all_joints_names[int(math.ceil(joint_pair[1] / 2))]
        start = arrows[joint_pair][0]
        end = arrows[joint_pair][1]
        b.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], head_width=3, head_length=6, fc=color_opt[id], ec=color_opt[id], label=end_joint_name)
        color_legend = mpatches.Patch(color=color_opt[id], label=end_joint_name)
        color_legends.append(color_legend)

    plt.legend(handles=color_legends, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def waitforbuttonpress():
    plt.waitforbuttonpress(timeout=1)