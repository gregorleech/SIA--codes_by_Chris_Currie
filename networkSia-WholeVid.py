#!/usr/bin/env python
# coding: utf-8

# # Spatial Image Autocorrelation Analysis Whole Video

# Import necessary files and whatnot. 
# $$$$ Remember to adjust the directory and the append path

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy
from scipy.ndimage import gaussian_filter1d as gf1d
from scipy.ndimage import gaussian_filter as gf
from scipy.ndimage import uniform_filter as uf
directory = "Z"
import sys
sys.path.append(directory+":\\Chris\\Code\\DDM\\")
import tiff_file
import ddm_clean as ddm
import io 
import sys
import csv
import os
import glob #glob is helpful for searching for filenames or directories
import pickle #for saving data
import array

import matplotlib.pyplot as plt
import seaborn as sns               #version 0.9.0
import warnings                        #version 0.4.1
import pandas as pd                 #version 0.23.4
                   #revision: 72223
import time
import datetime
import math
from collections import defaultdict                        #version 3.4.3
import scipy
from scipy import optimize
import scipy.special as sc


# # 1) Import
# $$$$$$$$
# Write the folder name that with your experiment in with this sort of style: Date_concentration type condition
# $$$$ Adjust the data_dir with the folder within Anderson lab network as given
# $$$$ Lastly, make the data_file equal to the name of the file you would like to analyze (check the printed statements to match what you are analyzing)
# $$$$ be sure to adjust the fps and the micron to pixel ratio

# In[2]:


"""210904_3.48 MT +0.04XL + 0.7 KN good(J) #graphs done, fits NOT done
210919_100 MT + 0.7 KN good(J) #no zstacks
210919_100 MT good(J)         #no zstacks
210919_4060 AMT + 0.7 KN + 0.47 MY + A-A good(s) #graphs done, fits NOT done
210919_4060 AMT + 0.7 KN + 0.47 MY + MT-MT Good(s) [Aggregate condesning]
210919_4060 AMT + 0.7 KN + A-A good(s)
210919_4060 AMT + 0.7 KN + MT-MT goodish (s) [Aggregate stuff]
210919_4060 AMT + 0.7 KN good(J)
210919_4060 AMT + 0.47 MY + A-A bad
210919_4060 AMT + 0.47 MY + MT-MT good"""


# In[10]:

'''
typee = "Kinesin\\"   # leave as an empty sting if Confocal videos isn't in a larger folder eg: Kinesin and add \\ at end if it is
exp = "211027_no XL + kinesin"          #folder name
foldy = ""                                      # leave as an empty sting if not needed
within_rt = ""     # Change to AT for kinesin paper exp
fps = 1
m2p_ratio = 0.8286                         # for microscope it is 0.828 micron to pixel ratio
#data_file = "1_nokinesin_zstack.nd2 - C=1"   #tif file
zstackframes = 5; do_all=0   # change depending on what you want for autocorrelation, minimum 3; do_all: 0=no and 1 = yes
data_dir = directory+":\\Chris\\"+typee+"Confocal Videos\\"+exp+foldy+"\\raw tiffs\\"+within_rt
saveto = directory+":\\Chris\\"+typee+"Confocal Videos\\"+exp+"\\SIA\\"  

files = glob.glob(data_dir+"*min*") 
files.sort()
print("found %i files" % len(files))
for i,f in enumerate(files): print (' %i \t %s' % (i, f.split('\\')[-1]))
    
found_file_number = 0'''


# In[26]:


cmap = matplotlib.cm.get_cmap('RdYlBu')
markerSize = 10
gf1dsize=2

def im_corr(image, filter=False, filtersize=256):
    if filter:
        image = image*1.0 - uf(image,filtersize)
    image = 1.0*image-image.mean()
    image = image/image.std()
    corr_im = abs(fftshift(ifft2(fft2(image)*np.conj(fft2(image)))))/(image.shape[0]*image.shape[1])
    rav_corr = ddm.newRadav(corr_im)
    return corr_im, rav_corr

def filtimage(image, filtersize=0):
    image = image*1.0 - uf(image,filtersize)
    return image

def powerlaw(t,A,C):
    return (A)*(t)**(C)

def exponential(t,A,k):
    return (A*np.exp(-k*t))

def exponent_par(xvalues,corr_ravs):
    exp_param = np.zeros((corr_ravs.shape[0],2))
    standerr = np.zeros((corr_ravs.shape[0],2))
    for i in range(corr_ravs.shape[0]):
        temp = scipy.optimize.curve_fit(exponential,xvalues[1:],corr_ravs[i][1:], absolute_sigma=True)
        exp_param[i,:] = temp[0]
        standerr[i,:] = np.sqrt(np.diag(temp[1]))/np.sqrt(corr_ravs.shape[0])
        #print(standerr,i)
        #print(corr_ravs.shape[0],corr_ravs.shape[1])
    return(exp_param,standerr)

def fit(param,x,y,function):
    for i in range(5,len(x)):
        res = y[i] - function(x[i],param[0],param[1])
        if res > 0:
            return i 
    return len(x)

def keyy(vid_len):
    if vid_len ==6:ikey = 5;
    elif 15<=vid_len <=18:ikey = 3;
    elif 20<=vid_len <=25:ikey = 4;
    elif 30<=vid_len <=36:ikey = 6;
    elif 42<=vid_len <=47:ikey = 8;
    return ikey

def average2Darr (corr_ravs,xvalues=[]):
    if xvalues == []:
        xvalues = corr_ravs[:,0]
        corr_ravs = corr_ravs[:,1:]
    summ = np.zeros(corr_ravs.shape[1]); talley = 0
    for i in range(corr_ravs.shape[1]):
        summ = summ + gf1d(corr_ravs[i],gf1dsize)
        talley += 1
    avg_corr = summ/talley
    plt.figure()
    plt.semilogx(xvalues,avg_corr)
    plt.show()

def average_fit(y):
    summ = 0
    for i in range(len(y)):
        summ = summ + y[i]
    avg_fit = summ/len(y)
    return avg_fit

def chip(xvalues,corr_ravs,finalFileLoc,ikey,c,vid_ti,chop = 30):
    x_dalues = np.linspace(0,chop,chop+1,dtype=int)
    '''plt.figure()
    plt.semilogx(xvalues,gf1d(corr_ravs[0],gf1dsize),'.',ms=markerSize,c=cmap(0),label="t = 0 s")
    for i in range(1,corr_ravs.shape[0]):
        plt.plot(xvalues, gf1d(corr_ravs[i],gf1dsize),'.',ms=markerSize,c=cmap(i/(corr_ravs.shape[0]-1)),
                 label=str(i*6)+" min")
    plt.title(finalFileLoc+" ("+vid_ti+", "+c+")")
    plt.xlabel("Distance ($\mu$m)",fontsize=9)
    plt.ylabel("Autocorrelation",fontsize=9)
    #plt.legend(loc=0,fontsize=7)
    plt.locator_params(axis='y', nbins=5)
    #plt.ylim(-0.04,0.9)
    plt.show()'''
    if ikey < 0: ikokay = -ikey
    else: ikokay = ikey
    corr_values = np.zeros((ikokay,len(x_dalues)))
    for i in range(len(x_dalues)):
        corr_values[:,i] = corr_ravs[:,i]
    corr_values = corr_values[:,1:]
    #print(corr_values)
    x_dalues = x_dalues[1:]
    print(x_dalues)
    return x_dalues

def compiler(exp,files,found_file_number,data_dir,saveto,fps,m2p_ratio,position_len=0):
    full_data_file= files[found_file_number]
    data_file = (full_data_file.split('\\')[-1])[:-4]
    #print(full_data_file)
    #print(data_file)

    def find_2nd(string, substring):
        return string.find(substring, string.find(substring) + 1)
    def find_1st(string, substring):
        return string.find(substring, string.find(substring))

    if fps == .36:
        finame = find_2nd(exp," ")
    else: finame = find_1st(exp," ")
    gibbb = find_1st(exp,"_")
    finalFileLoc = exp[gibbb+1: len(exp):1]
    vid_len = 6

    index = find_2nd(data_file,"_")
    findm = data_file.find("m"); find_hrs = 0; findz = -1; findx = -1
    if findm == -1: findh = data_file.find("h"); findm = findh; find_hrs = 1; '''vid_len = -1'''; findz = -1 # why vid len = -1?
    if findm == -1 and findh == -1:
        findz = data_file.find("z"); findm = -1; find_hrs = 0; vid_len = -1; findx = -1
        if findz == -1:
            findx = data_file.find("x");findm = -1; find_hrs = 0
        #print("z loc is",findz)
    if find_hrs ==1: fps == 10
    find_ = exp.find("_")
    if findm != -1:
        vid_len = float(data_file[index+1:findm:1])          #Comment out if you want just first 6 min
    vid_date = exp[0:find_:1]                     #Remeber to make a folder under each type of exp for date
    vid_ti = data_file[0:index:1]
    c = data_file[len(data_file)-3:len(data_file):1]
    if find_hrs == 1: un = "hrs";
    else: 
        if findm == -1:un = "frame"
        else: un = "min"
    print(un)
    print("This analysis will be in this folder: '" +finalFileLoc+"'")
    print("This analysis is of the video of length: '" +str(vid_len)+ "' " + un)
    print("This analysis is of the date of the video: '" +vid_date+ "'")

    #experiment names
    # 1) 20200625_5.8 AMT +myosin_568 actin + 488 mt
    # 2) 20200707_5.8 AMT_post contraction

    #file names
    # 1) 2_first 6min_continuous stimulation.nd2 - C=0
    #    2_first 6min_continuous stimulation.nd2 - C=1
    #    3_post 4hrs_488 stimulation.nd2 - C=0
    #    3_post 4hrs_488 stimulation.nd2 - C=1
    #    4_post 6min_continuous stimulation.nd2 - C=0
    #    4_post 6min_continuous stimulation.nd2 - C=1
    #    5_post  4hrs_no stimulation.nd2 - C=0

    # 2) 2_+488_6min.nd2 - C=0
    #    2_+488_6min.nd2 - C=1
    #    3_561_4hrs.nd2 - C=0
    #    5_+488_6min.nd2 - C=0
    #    5_+488_6min.nd2 - C=1
    #    6_561_4hrs.nd2 - C=0

    ims_dat = tiff_file.imread(data_dir+data_file+".tif");
    fwee = ims_dat.shape[0]
    #if do_all == 1: 
    bmw = fwee
    #else: bmw = zstackframes

    ikey = fwee
    length_frame = array.array('i',(i for i in range(0,fwee)))#np.linspace(0,fwee-1,fwee)
    #print(fwee,length_frame)
    fill = 98*3
    if fps == 1 or fps == 10:  # or fps == 2.65
        ims = tiff_file.imread(data_dir+data_file+".tif", key=length_frame);delta_f = ikey/fill

    delta_f = int(delta_f)
    print(delta_f)
    datapoints = np.zeros(fill+1, dtype=int)
    for i in range(fill+1):
        datapoints[i] = i*delta_f
    #print(datapoints)
    imsdata = tiff_file.imread(data_dir+data_file+".tif", key=datapoints);
    print("This will use the " +str(ikey)+ " key")
    print("Shape of ims is %i,%i,%i" % ims.shape)
    print("So %i frames of %ix%i pixels each" % ims.shape)
    fr_size = [ims.shape[1],ims.shape[2]];print(fr_size)
    
    '''print(len(ims))
    #if len(ims) > 1: 
    for i in range(len(ims)):
        plt.matshow(ims[i],cmap=cm.gray)
    #else: plt.matshow(ims[0],cmap=cm.gray)'''
    
    #plt.matshow(filtimage(ims[0],filtersize=1000))
    #plt.matshow(ims[0])
    plt.matshow(ims[0]-filtimage(ims[0],filtersize=256))
    
    corr_ims = np.zeros_like(ims[:,:,:])
    corr_ims_dat = np.zeros_like(imsdata[:,:,:])
    corr_ravs = np.zeros((ims.shape[0],100))
    corr_ravs_dat = np.zeros((imsdata.shape[0],100))
    filtersize=fr_size[0]
    for i in range(ims.shape[0]):
        corr_ims[i], temp = im_corr(ims[i,:,:], filter=True, filtersize=filtersize)
        corr_ravs[i] = temp[:corr_ravs.shape[1]]

    for i in range(imsdata.shape[0]):
        corr_ims_dat[i], temp = im_corr(imsdata[i,:,:], filter=True, filtersize=filtersize)
        corr_ravs_dat[i] = temp[:corr_ravs_dat.shape[1]]

    cmap = matplotlib.cm.get_cmap('RdYlBu')

    fig, ax = plt.subplots(figsize=(5,5/1.618))
    xvalues = np.arange(1,len(corr_ravs[0])+1) * m2p_ratio           # VERY IMPORTANT TO ADJUST THIS ABOVE
    ax.tick_params(axis='both', which='major', labelsize=7)
    if find_hrs == 0:
        if findm == -1:
            zilch = np.linspace(0,200,201); zilch = zilch.astype(int); minut = zilch.astype(str)
        elif vid_len < 12:
            minut = ["0","1","2","3","4","5","6","7","8","9","10","11"]
        else: minut = ["0","6","12","18","24","30","36","42","48"]
    elif find_hrs == 1:
        if 14<=vid_len <=16: #hours
            minut = ["0","3","5.5","8","11","14","16.5","19","22"]; contul = "_every_ab_3hrs"
        elif 5<=vid_len <=8: #hours
            minut = ["0","1","2","3","4","5","6","7","8"];contul = "_every_hour"
    markerSize = 10
    gf1dsize=2

    tweet = ims.shape[0]; print(tweet)

    for i in range(tweet):
        plt.semilogx(xvalues, gf1d(corr_ravs[i],gf1dsize),'.',ms=markerSize,c=cmap(i/(tweet-.75)),label="case "+str(i+1))

    plt.title(finalFileLoc+" ("+vid_ti+", "+c+")")
    plt.xlabel("Distance ($\mu$m)",fontsize=9)
    plt.ylabel("Autocorrelation",fontsize=9)
    #plt.legend(loc=0,fontsize=7)
    plt.locator_params(axis='y', nbins=5)
    plt.ylim(-0.04,1.01)
    #plt.xlim(-10,10)
    
    length = "_whole_vid"


    fig.savefig(saveto+"Graphs\\"+data_file+"_autocorrelation"+length+"_totframe_"+str(bmw)+"_whole_vid"+".png",dpi=600)

    data_vals = np.zeros((corr_ravs.shape[1],corr_ravs.shape[0]+1))
    for i in range (len(xvalues)):
        data_vals[i,:] = xvalues[i]
    for j in range(1,corr_ravs.shape[0]+1):
        data_vals[:,j] = corr_ravs[j-1,:]
    from numpy import asarray
    from numpy import savetxt
    from numpy import loadtxt


    savetxt(saveto+"Graphs\\"+data_file+"_autocorrelation_data"+length+"_whole_vid"+".csv",data_vals,delimiter=',')
    dataloaded = loadtxt(saveto+"Graphs\\"+data_file+"_autocorrelation_data"+length+"_whole_vid"+'.csv', delimiter=',')
    '''plt.figure()
    plt.semilogx(dataloaded[:,0],dataloaded[:,1],color='r')
    for i in range(2,dataloaded.shape[1]):
        plt.plot(dataloaded[:,0],dataloaded[:,i])

    fig.savefig(saveto+"Graphs\\"+data_file+"_autocorrelation_redownloaded"+length+"_whole_vid"+".png",dpi=600)
    plt.show()'''

    '''if vid_len != -1:
        x_val_dat = np.arange(len(corr_ravs_dat[0]))*m2p_ratio
        fig = plt.figure()
        plt.semilogx(x_val_dat,gf1d(corr_ravs_dat[0],gf1dsize),'.',ms=markerSize,c=cmap(0),label="t = 0 s")
        for i in range(1,corr_ravs_dat.shape[0]):
            plt.plot(x_val_dat, gf1d(corr_ravs_dat[i],gf1dsize),'.',ms=markerSize,c=cmap(i/fill),label=str()+'6 min')
        plt.title(finalFileLoc+" ("+vid_ti+", "+c+")")
        plt.xlabel("Distance ($\mu$m)",fontsize=9)
        plt.ylabel("Autocorrelation",fontsize=9)
        #plt.legend(loc=0,fontsize=7)
        plt.locator_params(axis='y', nbins=5)
        #plt.ylim(-0.04,0.9)
        plt.show()'''

    if vid_len != -1:
        fig.savefig(saveto+"\\Graphs\\"+data_file+"_autocorrelation_100_1ox"+length+"_whole_vid.png",dpi=600)

    pow_param = np.zeros((corr_ravs.shape[0],2))
    for i in range(corr_ravs.shape[0]):
        temp = scipy.optimize.curve_fit(powerlaw,xvalues[1:],corr_ravs[i][1:], absolute_sigma=True)
        pow_param[i,:] = temp[0]
        standerr = np.sqrt(np.diag(temp[1]))/np.sqrt(corr_ravs.shape[0])
    #print(pow_param)
    
    if ikey < 0: ikokay = -ikey
    else: ikokay = ikey
    if ikokay != 1: divis = ikokay-1
    else: divis = ikokay
    plt.figure()
    for i in range(ikokay):
        plt.loglog(xvalues,corr_ravs[i],'.',ms=markerSize,c=cmap(i/(divis)),label=str(i*6)+' min')
        plt.plot(xvalues,powerlaw(xvalues,pow_param[i,0],pow_param[i,1]),c=cmap(i/(divis)))
    plt.title("Power Law Fit of "+finalFileLoc+" ("+vid_ti+", "+c+")")
    plt.xlabel("Distance ($\mu$m)",fontsize=9)
    plt.ylabel("Autocorrelation",fontsize=9)
    #plt.legend(loc=0,fontsize=7)
    plt.show()

    fits = [0] * ikokay
    print(fits)
    for j in range (5,len(xvalues)):
        exp_param,standerr = exponent_par(xvalues[:j], corr_ravs[:,:j])
        plt.figure()
        for i in range(ikokay):
            temp_fit = fit(exp_param[i,:],xvalues[:j],corr_ravs[i,:j],exponential)
            if temp_fit > fits[i]:
                fits[i] = temp_fit
            elif findm == -1:
                plt.semilogy(xvalues[1:fits[i]+1],corr_ravs[i,1:fits[i]+1],'.',ms=markerSize,c=cmap(i/(divis)),label=un + " " + minut[i])
                plt.plot(xvalues[:j],exponential(xvalues[:j],exp_param[i,0],exp_param[i,1]),c=cmap(i/(divis)))
            elif find_hrs==0:
                plt.semilogy(xvalues[1:fits[i]+1],corr_ravs[i,1:fits[i]+1],'.',ms=markerSize,c=cmap(i/(divis)),label=str(i*6)+' min')
                plt.plot(xvalues[:j],exponential(xvalues[:j],exp_param[i,0],exp_param[i,1]),c=cmap(i/(divis)))
            else:
                plt.semilogy(xvalues[1:fits[i]+1],corr_ravs[i,1:fits[i]+1],'.',ms=markerSize,c=cmap(i/(divis)),label=minut[i]+" "+un)
                plt.plot(xvalues[:j],exponential(xvalues[:j],exp_param[i,0],exp_param[i,1]),c=cmap(i/(divis)))
            #plt.errorbar(xvalues[1:fits[i]+1],corr_ravs[i,1:fits[i]+1],yerr = standerr[i,1],fmt='o',c=cmap(i/(ikey-1)))
        plt.title("Exponential fit of "+finalFileLoc+" ("+vid_ti+", "+c+")")
        plt.xlabel("Distance ($\mu$m)",fontsize=9)
        plt.ylabel("Autocorrelation",fontsize=9)
        #plt.legend(loc=0,fontsize=7)
        plt.show()
        if fits[np.argmax(fits)] + 8 < j:
            break
    print(corr_ravs.shape[0])
    corr_values = []
    for i in range(corr_ravs.shape[0]):
        temp = corr_ravs[i,1:fits[i]+1]
        corr_values.insert(i, temp)
    print(corr_values)
    #mxlist, mxlen = FindMaxLength(corr_values)
    x_dalues = []
    for i in range(ikokay):
        xist = chip(xvalues,corr_ravs,finalFileLoc,ikey,c,vid_ti,len(corr_values[i]))
        print(xist)
        x_dalues.insert(i, xist)
    print(x_dalues)
    
    time = np.linspace(0,ikokay*6,ikokay)



    pow_param = np.zeros((len(corr_values),2))
    for i in range(len(corr_values)):
        temp = scipy.optimize.curve_fit(powerlaw,x_dalues[i],corr_values[i], absolute_sigma=True)
        pow_param[i,:] = temp[0]
    #print(pow_param)

    

    exp_param = np.zeros((len(corr_values),3))
    exp_param1 = np.zeros((len(corr_values),2))
    for i in range(len(corr_values)):
        temp = scipy.optimize.curve_fit(exponential,x_dalues[i],corr_values[i], absolute_sigma=True)
        standerr[i,:] = np.sqrt(np.diag(temp[1]))/np.sqrt(corr_ravs.shape[0])
        exp_param1[i,:] = temp[0]
    for i in range(len(corr_values)):
        for j in range(exp_param.shape[1]-1):
            exp_param[i,j] = exp_param1[i,j]
        exp_param[i,2] = standerr[i,1]
    print(exp_param1,exp_param,standerr)

    savetxt(saveto+"Fits\\"+data_file+"_autocorrelation_pow_fit_chopped_whole_vid.csv",pow_param,delimiter=',')
    savetxt(saveto+"Fits\\"+data_file+"_autocorrelation_exp_fit_chopped_whole_vid.csv",exp_param,delimiter=',')

    fig, ax = plt.subplots(figsize=(14,4))
    plt.subplot(1,2,1)
    for i in range(ikokay):
        if findz != -1:
            plt.loglog(x_dalues[i],corr_values[i],'.',ms=markerSize,c=cmap(i/(ikokay-1)),label=un + " " + minut[i])
            plt.plot(x_dalues[i],powerlaw(x_dalues[i],pow_param[i,0],pow_param[i,1]),c=cmap(i/(ikokay-1)))
        elif find_hrs ==0:
            plt.loglog(x_dalues[i],corr_values[i],'.',ms=markerSize,c=cmap(i/(ikey-1)),label=str(i*6)+' min')
            plt.plot(x_dalues[i],powerlaw(x_dalues[i],pow_param[i,0],pow_param[i,1]),c=cmap(i/(ikey-1)))
        else: 
            plt.loglog(x_dalues[i],corr_values[i],'.',ms=markerSize,c=cmap(i/(ikey-1)),label=minut[i]+" "+un)
            plt.plot(x_dalues[i],powerlaw(x_dalues[i],pow_param[i,0],pow_param[i,1]),c=cmap(i/(ikey-1)))
    plt.title("Power Law Fit of "+finalFileLoc+" ("+vid_ti+", "+c+")")
    plt.xlabel("Distance ($\mu$m)",fontsize=9)
    plt.ylabel("Autocorrelation",fontsize=9)
    #plt.legend(loc=0,fontsize=7)
    #plt.locator_params(axis='y', nbins=5)
    #plt.ylim(1e-3,1)
    plt.subplot(1,2,2)
    for i in range(ikokay):
        if findz != -1:
            plt.plot(x_dalues[i],corr_values[i] - powerlaw(x_dalues[i],pow_param[i,0],pow_param[i,1]),'.',ms=markerSize,
                         c=cmap(i/(ikokay-1)),label=un + " " + minut[i])
        elif find_hrs ==0:
            plt.plot(x_dalues[i],corr_values[i] - powerlaw(x_dalues[i],pow_param[i,0],pow_param[i,1]),'.',ms=markerSize,
                         c=cmap(i/(ikey-1)),label=str(i*6)+' min')
        else: plt.plot(x_dalues[i],corr_values[i] - powerlaw(x_dalues[i],pow_param[i,0],pow_param[i,1]),'.',ms=markerSize,
                         c=cmap(i/(ikey-1)),label=minut[i]+" "+un)
    plt.title("Power Law Residual of "+finalFileLoc+" ("+vid_ti+", "+c+")")
    plt.xlabel("Distance ($\mu$m)",fontsize=9)
    plt.ylabel("Autocorrelation",fontsize=9)
    #plt.legend(loc=0,fontsize=7)
    #plt.locator_params(axis='y', nbins=5)
    plt.show()

    fig.savefig(saveto+"Fits\\"+data_file+"_autocorrelation_pow_fit_chopped_whole_vid.png",dpi=600)
    
    fitter = []
    fig, ax = plt.subplots(figsize=(14,4))
    plt.subplot(1,2,1)
    for i in range(ikokay):
        if find_hrs==0:
            plt.semilogy(x_dalues[i],corr_values[i],'.',ms=markerSize,c=cmap(i/(divis)),label=str(i*6)+' min')
        else: plt.semilogy(x_dalues[i],corr_values[i],'.',ms=markerSize,c=cmap(i/(divis)),label=minut[i]+" "+un)
        plt.plot(x_dalues[i],exponential(x_dalues[i],exp_param[i,0],exp_param[i,1]),c=cmap(i/(divis)))
    plt.title("Exponential fit of "+finalFileLoc+" ("+vid_ti+", "+c+")")
    plt.xlabel("Distance ($\mu$m)",fontsize=9)
    plt.ylabel("Autocorrelation",fontsize=9)
    #plt.legend(loc=0,fontsize=7)
    #plt.locator_params(axis='y', nbins=5)
    #plt.ylim(1e-3,1)
    plt.subplot(1,2,2)
    for i in range(ikokay):
        """if findz != -1:
            plt.plot(x_dalues[i],corr_values[i] - exponential(x_dalues[i],exp_param[i,0],exp_param[i,1]),'.',ms=markerSize,
                         c=cmap(i/(divis)),label=un+" "+minut[i])"""
        '''elif find_hrs==0 and findm==-1:'''
        plt.plot(x_dalues[i],corr_values[i] - exponential(x_dalues[i],exp_param[i,0],exp_param[i,1]),'.',ms=markerSize,
                 c=cmap(i/(ikey-1)),label=str(i*6)+' min')
        '''else:
            plt.plot(x_dalues[i],corr_values[i] - exponential(x_dalues[i],exp_param[i,0],exp_param[i,1]),'.',ms=markerSize,
                         c=cmap(i/(ikey-1)),label=minut[i]+" "+un)'''
    plt.title("Exponential Residual of "+finalFileLoc+" ("+vid_ti+", "+c+")")
    plt.xlabel("Distance ($\mu$m)",fontsize=9)
    plt.ylabel("Autocorrelation",fontsize=9)
    #plt.legend(loc=0,fontsize=7)
    #plt.locator_params(axis='y', nbins=5)
    plt.show()

    fig.savefig(saveto+"Fits\\"+data_file+"_autocorrelation_exp_fit_chopped_whole_vid.png",dpi=600)
    ari = np.zeros((len(x_dalues[-1]),2))
    ari[:,0] = np.array(x_dalues[-1])
    ari[:,1] = corr_values[-1]
    #ari = np.array(ari)
    print(ari)
    #print(corr_values)
    
    savetxt(saveto+"\\Fits\\"+data_file+"_autocorrelation_exp_fit_whole_vid"+"_whole_vid"+".csv",ari ,delimiter=',')
    
    average2Darr(corr_ravs, xvalues)
    avg_fit = average_fit(exp_param)
    print(avg_fit)
    
    return position_len, avg_fit[1], avg_fit[2]
#arr_b = np.zeros(3)    
#arr_b[0],arr_b[1],arr_b[2] = compiler(files,found_file_number,data_dir,saveto,fps,m2p_ratio)


