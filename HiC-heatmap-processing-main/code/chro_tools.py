#!/usr/bin/env python
# coding: utf-8

# In[1]:


import contig


# In[11]:


import chro_use as cu


# In[3]:


import column


# In[3]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import re
import json
import numpy as np
import pandas as pd
import cooler
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import chromosight.utils.contacts_map as mapp
from IPython.display import display
import chromosight.utils.io as io
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.signal import find_peaks
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.transform import resize
from skimage import data, img_as_float
import cv2
from operator import itemgetter
from findpeaks import findpeaks
from scipy import signal
from hic2cool import hic2cool_convert
import networkx as nx


# In[2]:



# In[5]:


def l_n_scatter(dic,all_dic,info):
    ne_dic = {}
    for i in range(13):
        new_list = []
        for key,values in dic.items():
            #generate aditionary with key = number of peaks been detected,value = contig length
            if len(values) == i:            
                new_list.append(get_length(all_dic.get(key),all_dic,info,'length'))
            ne_dic[i] = new_list
    
    fig, ax = plt.subplots()

    for x, ys in ne_dic.items():
        try:
            ax.scatter([x] * len(ys), ys,6)
        except ValueError:  #raised if `y` is empty.
            pass

    plt.show()





# In[ ]:


def peak_plot(dic):
       
    list_p = []
       
    for i in dic.values():
        length = len(i)
        list_p.append(length)
    new_dic = {}
    for j in range(35):
        count = 0
        for i in list_p:
            if i == j:
                count = count + 1
        new_dic[j] = count
   
    lists = sorted(new_dic.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    plt.plot(x, y)
    plt.xlabel('number of peaks')
    plt.ylabel('number of contigs')

    plt.show()
    return new_dic


# In[ ]:

# These three functions below use to image processing,not useful at this stage
def sift(gray,img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    img = cv2.drawKeypoints(gray,kp,img)
    return img
def show(src):
    src = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
    plt.imshow(src)
def threshold(img,x):
    ret, thresh = cv2.threshold(img,x,255,cv2.THRESH_BINARY)
    return thresh

#Another peak detection method,but perform not well
def get_ids_cwd(matrix,big_matrix,dic,choro_frame):
    matrix_mean = np.mean(matrix,axis = 0)
    median_height = np.median(smooth_scaffold(matrix_mean))
    peakind = signal.find_peaks_cwt(matrix_mean, np.arange(1,150))
    iddd = get_contig_id(matrix,dic)
    for i in peakind:
        idd = find_location(big_matrix,choro_frame,dic,i)
        print(iddd ,i, idd)


# In[6]:

#will generate a network graph
def net_graph(dic,label = False,font_size = 5):
    
    a_dic = {}
    for key,values in dic.items():
        a_list = []
        new_key = key[0:-12]
        for i in values:
            #remove not important string in contig id
            i = i[0:-12]
            a_list.append(i)
        a_dic[new_key] = a_list
    
    
    G = nx.Graph()
    G.add_nodes_from(a_dic.keys())
    for key,values in a_dic.items():
        #avoid bug when calculate vote_strength
        if len(values) == 1:
            continue
        #calculate weight, could add to nx.draw_spring()
        vote_strength = 1/(len(values)-1)
        for contig in values:
            if contig != key:
                G.add_edge(key,contig)
    nx.draw_spring(G,node_size = 5,with_labels = label,font_size = font_size )


# In[ ]:


def generate_contig_peak(matrix,dic,chro_info,all_dic):
    idd = get_contig_id(matrix,all_dic) 
    result = dic.get(idd)
    for i in result:
        matrixs = all_dic.get(i)
        start = get_length(matrixs,all_dic,chro_info,dtype = 'bin')[0]
        if matrix.shape[0] < 200:
            matrix = resize(matrix, (250, matrix.shape[1]))
            plt.imshow(matrix, cmap='afmhot_r',norm=colors.PowerNorm(gamma=0.15))
        else:
            plt.imshow(matrix ** 0.02, cmap='afmhot_r',norm=colors.PowerNorm(gamma=0.45))
        ioa = get_length(matrixs,all_dic,chro_info,dtype = 'length')
        plt.plot(start,125,marker='.',color = 'green')
        plt.axis('off')
        plt.title(idd,' ',i,' ' ,str(ioa))
        plt.show() 


# In[7]:


def generate_contig_peak_m(matrix,dic,chro_info,all_dic):
    idd = get_contig_id(matrix,all_dic) 
    result = dic.get(idd)
    start_l = []
    for i in result:
        matrixs = all_dic.get(i)
        start = get_length(matrixs,all_dic,chro_info,dtype = 'bin')[0]
        start_l.append(start)
        print(i)
    if matrix.shape[0] < 200:
        matrix = resize(matrix, (250, matrix.shape[1]))
        plt.imshow(matrix, cmap='afmhot_r',norm=colors.PowerNorm(gamma=0.15))
    
    else:
        plt.imshow(matrix*0.02, cmap='afmhot_r',norm=colors.PowerNorm(gamma=0.45))
    plt.title(idd)
    #plot in same graph
    for i in start_l:
        plt.plot(i,125,marker='.',color = 'green')
        plt.axis('off')
    plt.show()


# In[4]:

#only work for filter method,otherwise some detected peak will have none value because we already filter some contig rows
def optimize_result_af(dic):
    nnd={}
    
    for key,values in dic.items():
        listt = []
        for i in values:
            result = dic.get(i)
            if key in result:
                listt.append(i)
        nnd[key] = listt
    return nnd
                


# In[22]:


def get_contig_id(matrix,my_dict):
    for key, value in my_dict.items():
        if np.array_equal(value,matrix):
            return key
 
    return "key doesn't exist"


# In[17]:


def get_length(contig,dic,chro_info,dtype = 'bin'):
    z = chro_info.iloc[:,0]
    z = z.values.tolist()
    bin_loc = chro_info.iloc[:,2:4] #get start_bin and end_bin dataframe
    start_bin = bin_loc.iloc[:,0]# change to list
    end_bin = bin_loc.iloc[:,1]
    pp = 0
    for i in range(len(z)):
        if z[i] == str(get_contig_id(contig,dic)):
            pp = i
    length_list = chro_info.iloc[:,1].values.tolist()#length list
    if dtype == 'length':
        return length_list[pp]
    elif dtype == 'bin':
        return start_bin[pp],end_bin[pp]
    else:
        return z[pp]


# In[78]:


def merge_contig(id_list,info,all_dic):
    new_list = []
    new_list1 = []
    length = len(id_list)
    for i in range(length):
        matrix = all_dic.get(id_list[i])
        c_length = get_length(matrix,all_dic,info)
        new_list.append(matrix)
        new_list1.append(c_length)
    #concatenate contig row firstly
    labels =np.concatenate(new_list, axis=0)
    another_list = []
    #extract specfic region then concatenate
    for i in new_list1:
        x = labels[:,i[0]:i[1]]
        another_list.append(x)
    labelss =np.concatenate(another_list, axis=1)
    plt.imshow(labelss ** 0.02, cmap='afmhot_r',norm=colors.PowerNorm(gamma=0.45))
    
    

