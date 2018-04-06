# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:54:47 2016
@author: Kostya S, nms code from: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/ and
    https://github.com/layumi/2015_Face_Detection/blob/master/nms.m
"""

import scipy as sp
import itertools
import os
from fnmatch import fnmatch

class Util():
    k_max = 48
    

    sn = (0.83, 0.91, 1.0, 1.10, 1.21)
    xn = (-0.17, 0.0, 0.17)
    yn = (-0.17, 0.0, 0.17)
    prod = sp.array([e for e in itertools.product(sn,xn,yn)])
    
    @staticmethod
    def load_from_npz(data_name):
        with sp.load(data_name) as f:
            values = [f['arr_%d' % i] for i in range(len(f.files))][0]
            return values
        
    @staticmethod
    def get_files(rootdir = '',fexpr = ''):
        paths = []
        for path, subdirs, files in os.walk(rootdir):
            for indx,fname in enumerate(files):
                if fexpr == None or fexpr == '' or fnmatch(fname, fexpr):
                    paths.append(os.path.join(path, fname))
        return paths
                
    @staticmethod
    def get_splited_files(paths,batch_size = 2):
        splited_files = paths
        N = max(1,batch_size)
        for item in [splited_files[i:i + N] for i in range(0, len(splited_files), N)]:
            labels = [0 if e.find('neg') > 0 else 1 for e in item]
            yield item,labels
        
    
    @staticmethod
    def calib (i,j,h,w,n):
        prod = Util.prod
        new_i = round(i-(prod[n][1]*w/prod[n][0]))
        new_j = round(j-(prod[n][2]*h/prod[n][0]))
        new_h = round(h/prod[n][0])
        new_w = round(w/prod[n][0])
        return (new_i,new_j,new_h,new_w)
        
    @staticmethod
    def calib_apply(i,j,h,w,sn,xn,yn):
        
        new_i = round(i-(xn*w/sn))
        new_j = round(j-(yn*h/sn))
        new_h = round(h/sn)
        new_w = round(w/sn)
        return (new_i,new_j,new_h,new_w)
        
    
    @staticmethod
    def inv_calib (i,j,h,w,n):
        prod = Util.prod
        new_i = round(i-(-prod[n][1]*w/(prod[n][0]**-1)))
        new_j = round(j-(-prod[n][2]*h/(prod[n][0]**-1)))
        new_h = round(h/prod[n][0]**-1)
        new_w = round(w/prod[n][0]**-1)
        return (new_i,new_j,new_h,new_w)

    @staticmethod
    def nms(dets,proba, T):
        
        dets = dets.astype("float")
        if len(dets) == 0:
            return []
        
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = proba
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = sp.maximum(x1[i], x1[order[1:]])
            yy1 = sp.maximum(y1[i], y1[order[1:]])
            xx2 = sp.minimum(x2[i], x2[order[1:]])
            yy2 = sp.minimum(y2[i], y2[order[1:]])
        
            w = sp.maximum(0.0, xx2 - xx1 + 1)
            h = sp.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = sp.where(ovr <= T)[0]
            order = order[inds + 1]
        
        return keep

    @staticmethod 
    def minmax(arr):
        minv = arr[0]
        maxv = arr[0]
        for val in arr:
            if val > maxv:
                maxv = val
            elif val < minv:
                minv = val
        return int(sp.floor(minv)), int(sp.ceil(maxv))    
    
    @staticmethod 
    def ellipse2bbox(a, b, angle, cx, cy):
        a, b = max(a, b), min(a, b)
        ca = sp.cos(angle)
        sa = sp.sin(angle)
        if sa == 0.0:
            cta = 2.0 / sp.pi
        else:
            cta = ca / sa
    
        if ca == 0.0:
            ta = sp.pi / 2.0
        else:
            ta = sa / ca
    
        x = lambda t: cx + a * sp.cos(t) * ca - b * sp.sin(t) * sa
    
       
        y = lambda t: cy + b * sp.sin(t) * ca + a * sp.cos(t) * sa
    
        # x = cx + a * cos(t) * cos(angle) - b * sin(t) * sin(angle)
        # tan(t) = -b * tan(angle) / a
        tx1 = sp.arctan(-b * ta / a)
        tx2 = tx1 - sp.pi
        x1, y1 = x(tx1), y(tx1)
        x2, y2 = x(tx2), y(tx2)
    
        # y = cy + b * sin(t) * cos(angle) + a * cos(t) * sin(angle)
        # tan(t) = b * cot(angle) / a
        ty1 = sp.arctan(b * cta / a)
        ty2 = ty1 - sp.pi
        x3, y3 = x(ty1), y(ty1)
        x4, y4 = x(ty2), y(ty2)
    
        minx, maxx = Util.minmax([x1, x2, x3, x4])
        miny, maxy = Util.minmax([y1, y2, y3, y4])
        return sp.floor(minx), sp.floor(miny), sp.ceil(maxx), sp.ceil(maxy)

    @staticmethod
    def kfold(X,y, k_fold = 2):
        for k in range(k_fold):
            t_indx = [i for i, e in enumerate(X) if i % k_fold != k]
            v_indx = [i for i, e in enumerate(X) if i % k_fold == k]
            yield t_indx,v_indx