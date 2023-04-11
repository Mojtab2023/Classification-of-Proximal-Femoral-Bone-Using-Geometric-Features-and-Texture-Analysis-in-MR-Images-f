# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:02:31 2022

@author: Marab
"""
Address='C:/Users/Marab/Desktop/Osteoprosis_detection/Dataset/Femur_segmented/'
#Address should point to the directory of images that we aim to extract features from
FV_Address='H:/Projects_and_works/Finished Projects/MRI_Imaging/Revisions of the paper/FV_Paper.xlsx'
#FV_Address is the Address you want to save the feature vector (FV_paper.xlsx) in
# import Libraries
import xlsxwriter
import os

import copy
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix,graycoprops
from skimage.transform import radon, rescale
import plantcv as pcv
from scipy import ndimage
from scipy.stats import norm, kurtosis,skew,moment
global thrsh
thrsh=0

plt.close('all')
def grid_mass_FE(img,rows,cols):
    #partition the image into grids with rows*cols size and sum up the pixels
    s, v= img.shape
    FV1=[]
    for i in range(int(s/rows)-1):
        for j in range(int(v/cols)-1):                       
            FV1.append(np.sum(img[i*rows:(i+1)*rows,j*cols:(j+1)*cols]))
    FV1=(FV1-np.amin(FV1))/(np.amax(FV1)-np.amin(FV1))#normalize the vector elemnts in 0 to 1
    return FV1
def clip_effective_part(img):
    y,x=img.shape
    for i in range(y):
        if np.sum(img[i,:])>0:
            start_h=i
            break
    for i in range(y-1,0,-1):
        if np.sum(img[i,:])>0:
            stop_h=i
            break
    for i in range(x):
        if np.sum(img[:,i])>0:
            start_v=i
            break
    for i in range(x-1,0,-1):
        if np.sum(img[:,i])>0:
            stop_v=i
            break
    return img[start_h:stop_h,start_v:stop_v]
def clip_effective_part_aswise(img,img_aswise):
    y,x=img.shape
    for i in range(y):
        if np.sum(img[i,:])>0:
            start_h=i
            break
    for i in range(y-1,0,-1):
        if np.sum(img[i,:])>0:
            stop_h=i
            break
    for i in range(x):
        if np.sum(img[:,i])>0:
            start_v=i
            break
    for i in range(x-1,0,-1):
        if np.sum(img[:,i])>0:
            stop_v=i
            break
    return img[start_h:stop_h,start_v:stop_v],img_aswise[start_h:stop_h,start_v:stop_v]
def skel_prun(skel,target):
    skel=clip_effective_part(skel)
    skel_pruned=copy.copy(skel)
    y,x=skel.shape
    cnt=0
    while cnt<target:
        for i in range(y):
            for j in range(x):
                if skel[i,j]==1 and np.sum(skel[i-1:i+2,j-1:j+2])<3:
                    skel_pruned[i,j]=0
                    cnt +=1
        skel=copy.copy(skel_pruned)
    skel=clip_effective_part(skel)
    return skel           

def geo_FE(img):
    # perform skeletonization
    #skeleton = skeletonize(Mask/255)
    Mask=img>thrsh
    Mask=Mask
    skeleton = pcv.morphology.skeletonize(mask=Mask)
    pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skel_img=skeleton, size=100)
    s, v= pruned_skeleton.shape
    #-----find coordinates of the corner and two other points to measure the angle
    pruned_skeleton=pruned_skeleton/255
    
    for cY in range(s-1,0,-1):
        if np.sum(pruned_skeleton[cY,:])>1:
            print("here",cY)
            for cX in range(v):
                if pruned_skeleton[cY,cX]==1:
                    print("here2",cX)
                    Cent_point=np.array([cY,cX])
                    break
            break
    for lX in range(v):
        if np.sum(pruned_skeleton[:,lX])>0:
            for lY in range(s-1,0,-1):
                if pruned_skeleton[lY,lX]==1:
                    Left_point=np.array([lY,lX])
                    break
            break
    for lY in range(s-1,0,-1):
        if np.sum(pruned_skeleton[lY,:])>0:
            for lX in range(v):
                if pruned_skeleton[lY,lX]==1:
                    Lower_point=np.array([lY,lX])
                    break
            break
    vec1=(Cent_point-Lower_point)
    vec1_size=np.linalg.norm(vec1)
    vec1=vec1/vec1_size# normalization of vectors
    vec2=(Cent_point-Left_point)
    vec2_size=np.linalg.norm(vec2)
    vec2=vec2/vec2_size
    cosine_angle = np.dot(vec1, vec2) 
    angle = np.arccos(cosine_angle)
    wvb1=np.sum(Mask[Lower_point[0],:])#width of vertical bone
    wvb2=np.sum(Mask[Lower_point[0]-50,:])#width of vertical bone
    
    cosbt=Left_point+(Cent_point-Left_point)/2#center of slopy bone
    cosb=[int(cosbt[0]), int(cosbt[1])]
    dist=[]
    for i in range(cosb[0]):
        for j in range(v-1,cosb[1],-1):
            if Mask[i,j]==0:
                dist.append(np.linalg.norm([cosb[0]-i,cosb[1]-j]))
                if dist[-1]==np.min(dist):
                    Yn1=copy.copy(i)
                    Xn1=copy.copy(j)                
    dist=[]
    for i in range(s-1,cosb[0],-1):
        for j in range(cosb[1]):
            if Mask[i,j]==0:
                dist.append(np.linalg.norm([cosb[0]-i,cosb[1]-j]))
                if dist[-1]==np.min(dist):
                    Yn2=copy.copy(i)
                    Xn2=copy.copy(j)                
    whb=np.linalg.norm([Yn1-Yn2,Xn1-Xn2])              
    
    GEO_FV=np.array([vec1_size,vec2_size,angle,wvb1,wvb2,wvb1/wvb2,whb])      
    return skeleton,pruned_skeleton, segmented_img, segment_objects,GEO_FV  

def virticalize(img0):
    img=copy.copy(img0)
    img=img>thrsh
    img=clip_effective_part(img)
    y,x=img.shape
    
    for i in range(x):
        if img[y-20,i]>0:
            BL=i#Bottom left
            break
    for i in range(x):
        if img[y-70,i]>0:
            UL=i#Up left
            break
    for i in range(x-1,0,-1):
        if img[y-20,i]>0:
            BR=i
            break
    for i in range(x-1,0,-1):
        if img[y-70,i]>0:
            UR=i
            break
    Up=(UR+UL)/2
    BT=(BR+BL)/2
    if Up-BT>0:
        Angle=90-np.arctan([50/(np.abs(Up-BT))])*180/np.pi
    else:
        Angle=np.arctan([50/(np.abs(Up-BT))])*180/np.pi-90
    img0 = ndimage.rotate(img0, int(Angle))
    img0=clip_effective_part(img0) 
    return Angle,img0

def indexing(img0):    
    img=img0>5
    img,img0=clip_effective_part_aswise(img,img0)
    indexed_img0=copy.copy(img0)
    y,x=img.shape
   
    ## ===============Finding the circle center on the left corner of the image============
    for i in range(x):
        if img[0,i]>0:
            col1=i
            break
    for i in range(x-1,0,-1):
        if img[0,i]>0:
            col2=i
            break
    col=int((col1+col2)/2)    
    
    #find lower side of the circle as its diameter
    for i in range(y-1,0,-1):
        if np.sum(img[i,0:col])>0:
            row=int(i/2)
            break
    Cent_point_1=[row,col]
    indexed_img0[row-1:row+2,col-1:col+2]=255    
    
    ## ==============Finding right part center==========
    for i in range(y):
        if img[i,x-1]>0:
            row_Up=i
            break
    for i in range(y-1,0,-1):
        if img[i,x-1]>0:
            row_Down=i
            break
    row_Rightside=int((row_Up+row_Down)/2)
    indexed_img0[row_Rightside,:]=255
    
    for i in range(x):
        if img[y-20,i]>0:
            BL=i#Bottom left
            break
    for i in range(x-1,0,-1):
        if img[y-20,i]>0:
            BR=i
            break
    Vertical_cent=int((BL+BR)/2)
    indexed_img0[:,Vertical_cent]=255
    
    Cent_point_2=[row_Rightside,Vertical_cent]
    
    ## find upper right boundary curve and devide it into 2 sections, right of tha horn and left of it   =========
    curve1=np.zeros([1,x-1-Cent_point_1[1]])
    for i in range(x-1,Cent_point_1[1],-1):
        for j in range(y):
            if img[j,i]==0:
                curve1[0,x-i-1]=curve1[0,x-i-1]+1
            else:
                break    
    curve1_intersec=x-Cent_point_2[1]
    horn_ind=np.argmin(curve1[0][0:curve1_intersec])
    curve1_1=curve1[0][:horn_ind]
    curve1_2=curve1[0][horn_ind:]   
    ## find upper left boundary curve, left of the circle=========================================================
    curve1_0=np.zeros([1,Cent_point_1[1]])
    for i in range(0,Cent_point_1[1]):
        for j in range(y):
            if img[j,i]==0:
                curve1_0[0,i]=curve1_0[0,i]+1
            else:
                break    
    FV_curve1_0=simple_stat_FE(curve1_0[0])
    ## find left boundary curve, below the circle=========================================================
    curve2_0=np.zeros([1,y-Cent_point_1[0]])
    for i in range(Cent_point_1[0],y):
        for j in range(x):
            if img[i,j]==0:
                curve2_0[0,i-Cent_point_1[0]]=curve2_0[0,i-Cent_point_1[0]]+1
            else:
                break 
    ## find right boundary curve, below the circle=========================================================
    curve4_0=np.zeros([1,y-Cent_point_2[0]])
    for i in range(Cent_point_2[0],y):
        for j in range(x-1,0,-1):
            if img[i,j]==0:
                curve4_0[0,i-Cent_point_2[0]]=curve2_0[0,i-Cent_point_2[0]]+1
            else:
                break                
            
    FV_curve1_1=simple_stat_FE(curve1_1)
    FV_curve1_2=advanced_stat_FE(curve1_2)        
    FV_curve1_0=simple_stat_FE(curve1_0[0])
    FV_curve2_0=simple_stat_FE(curve2_0[0])
    FV_curve4_0=simple_stat_FE(curve4_0[0])
    
    Teta=int(90+np.arctan([(Cent_point_2[0]-Cent_point_1[0])/(Cent_point_2[1]-Cent_point_1[1])])*180/np.pi)
    img_temp = ndimage.rotate(img0, int(Teta-90))>10
    img_temp=clip_effective_part(img_temp)    
    Neck_withs=np.min(np.sum(img_temp[:,Cent_point_1[1]:Cent_point_2[1]],axis=0))    
    try:
        Sagh_withs1=np.sum(img_temp[Cent_point_2[0]+20,:])
    except:
        Sagh_withs1=0
    try:
        Sagh_withs2=np.sum(img_temp[Cent_point_2[0]+40,:])
    except:
        Sagh_withs2=0
    try:
        Sagh_withs3=np.sum(img_temp[Cent_point_2[0]+60,:])
    except:
        Sagh_withs3=0
    
    ##distance of Cent_point_2 to outer bound of the image      
    for i in range(Cent_point_2[0]):
        for j in range(x-1,Cent_point_2[1],-1):        
            if img[i,j]==1:
                far_point=[i,j]                
                break
        else:
            continue
        break
    ###=========proposal based features==============
    proposal_based_features1=[np.mean(img0),np.var(img0),skew(img0,axis=None)]
    #https://gist.github.com/rougier/e5eafc276a4e54f516ed5559df4242c0
    proposal_based_features2=[fractal_dimension(img0)]#fractal dimensions
    #https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html
    g = graycomatrix(img0.astype(np.uint8), [1, 2], [0, np.pi/2], levels=256,normed=True, symmetric=True)
    contrast = graycoprops(g, 'contrast')
    proposal_based_features3=contrast.flatten()#Coocurance based features (GLCM)
    ###-----------------------------------------------
    UR_farpoint=np.linalg.norm([far_point[0]-Cent_point_2[0],far_point[1]-Cent_point_2[1]])
    x,y=img.shape
    FV=[Cent_point_1[0],Cent_point_1[1],np.linalg.norm([Cent_point_2[0]-Cent_point_1[0],
                                                        Cent_point_2[1]-Cent_point_1[1]]),
        Teta,Neck_withs,UR_farpoint,Sagh_withs1,Sagh_withs2,Sagh_withs3,x,y] 
    
    FV.extend(FV_curve1_1)
    FV.extend(FV_curve1_2)
    FV.extend(FV_curve1_0)
    FV.extend(FV_curve2_0)
    FV.extend(FV_curve4_0)
    
    FV.extend(list(proposal_based_features1))
    FV.extend(list(proposal_based_features2))
    FV.extend(list(proposal_based_features3))
    
    FV=np.asarray(FV)
    
    a=np.power(FV, 2)    
    b=np.power(FV, 3)
    
    FV=list(FV)
    FV.extend(a)
    FV.extend(b)
    print("Size of feature vector:",len(FV))
    return indexed_img0,FV,img0  

def ramped_remov_FE(in_sig):
    in_sig[in_sig != 0]#remove zero elements from the array
    Ramped=in_sig*0
    m=(in_sig[-1]-in_sig[0])/len(in_sig)
    t=np.linspace(0, len(in_sig), num=len(in_sig)).astype(int)
    Ramped=in_sig[0]+m*t   
    return in_sig-Ramped
def simple_stat_FE(in_sig):
    ramped_remov_FE(in_sig)
    in_sig[in_sig != 0]#remove zero elements from the array
    return [min(in_sig),max(in_sig),(max(in_sig)-min(in_sig))/len(in_sig),
            kurtosis(np.abs(ramped_remov_FE(in_sig)), fisher=True),skew(np.abs(ramped_remov_FE(in_sig))),
            (in_sig[-1]-in_sig[0])/len(in_sig),moment(ramped_remov_FE(in_sig), moment=1),
            moment(ramped_remov_FE(in_sig), moment=2),np.var(in_sig),np.max(np.abs(np.gradient(np.gradient(in_sig))))]
def advanced_stat_FE(in_sig):
    in_sig[in_sig != 0]#remove zero elements from the array
    return [min(in_sig),max(in_sig),np.mean(in_sig),np.var(in_sig),kurtosis(in_sig, fisher=True),skew(in_sig),
                 moment(in_sig, moment=1),moment(in_sig, moment=2)]
def sift_FE(img):
    sift = cv.xfeatures2d.SIFT_create()    
    keypoints,descriptor= sift.detectAndCompute(img, None)
    return
def fractal_dimension(Z, threshold=0.9):

    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Transform Z into a binary array
    Z = (Z < threshold)
    
    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

## =========================================================
workbook = xlsxwriter.Workbook(FV_Address)
worksheet = workbook.add_worksheet()    
T_FV=[]

A=os.listdir(Address)
cnt=0;
for i in A:
    print(i)
    img = cv.imread(Address+i, cv.IMREAD_GRAYSCALE)       
    img0,FV0,img00=indexing(img) 
    
    FV=FV0
    #FV=np.concatenate((FV0, FV1))
    
    col=0     
    worksheet.write(cnt+1,0,i)#write the number of the sample in the first column
    for items in FV:
        worksheet.write(cnt+1,col+1,items)
        col += 1 
    cnt+=1
col=0
for i in range(len(FV)):
    worksheet.write(0,col+1,'F'+str(col))
    col += 1
    
workbook.close()
