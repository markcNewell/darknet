# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

from scipy.misc import imread
import cv2

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im

def detect2(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    num = dn.c_int(0)
    pnum = dn.pointer(num)
    dn.predict_image(net, im)
    dets = dn.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): dn.do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    #free_image(im)
    #free_detections(dets, num)
    return res

import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import os

print(os.path.exists("model/model.weights"))

# Darknet
net = dn.load_net(b'model/model.cfg', b'model/model.weights', 0)
meta = dn.load_meta(b'cfg/obj.data')
#r = dn.detect(net, meta, b'data/frame.png')
#print(r)

# OpenCV
arr = cv2.imread('data/frame.png')
im = array_to_image(arr)
dn.rgbgr_image(im)
r = detect2(net, meta, im)
print(r)

