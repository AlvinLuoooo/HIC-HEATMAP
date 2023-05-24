#!/usr/bin/env python
# coding: utf-8

# In[2]:


import chro_use  as cu
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
from scipy.signal import find_peaks
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.transform import resize
from skimage import data, img_as_float


# In[ ]:





# In[6]:


class column:
    def __init__(self,chro_info,splited_matrix):
        self.splited_matrix = splited_matrix
        self.chro_info = chro_info
        
    #change window_length if resolution change
    def smooth_scaffold(self,scaffold, window_length=10):
        return np.convolve(scaffold, np.ones(window_length)/window_length, mode='same')
        
    def get_peak_dic(self):
        dic_n = {}
        for i in self.splited_matrix:
            i_mean = np.mean(i,axis = 0)
            median_height = np.median(self.smooth_scaffold(i_mean))
            #from scipy.signal.find_peaks
            peaks, _ = find_peaks(self.smooth_scaffold(i_mean), height = median_height*4, distance = 110)
            dic_n[get_contig_id(i,fish_dic)] = peaks
        return dic_n,peaks
    
    # generate peak plot graph,only work form column method,suggest use peak_plot function in chro_tools module
    def peak_plot(self,th = 4):
        list_p = []
        for i in self.splited_matrix:
            i_mean = np.mean(i,axis = 0)
            median_height = np.median(self.smooth_scaffold(i_mean))
            peaks, _ = find_peaks(self.smooth_scaffold(i_mean), height = median_height*th, distance = 110)
            list_p.append(len(peaks))
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
        plt.show()
        return new_dic
    
    # generate peak graph,only work form column method,suggest use generate_peak function in chro_tools module
    def generate_peak(self,matrix,all_dic, th = 4):
        matrix_mean = np.mean(matrix,axis = 0)
        median_height = np.median(self.smooth_scaffold(matrix_mean))
        peaks, _ = find_peaks(self.smooth_scaffold(matrix_mean), height = median_height*th, distance = 110,plateau_size = 1)
        right = _.get('right_edges')
        for i in peaks:
            idd = self.get_contig_id(matrix,all_dic)
            image = matrix
            if image.shape[0] < 200:
                image = resize(image, (250, image.shape[1]))
            plt.imshow(image ** 0.02, cmap='afmhot_r',norm=colors.PowerNorm(gamma=0.45))
            plt.plot(i,125,marker='.',color = 'green')
            plt.axis('off')
            plt.title(str(idd) + ', ' + str(i))
            plt.show()
            
    # get contig id of peak      
    def get_ids(self,matrix,big_matrix,all_dic,choro_frame,s_distance = 110,mode = None,value = 4,f_value = 500000):
        matrix_mean = np.mean(matrix,axis = 0)
        median_height = np.median(self.smooth_scaffold(matrix_mean))
        peaks, _ = find_peaks(self.smooth_scaffold(matrix_mean), height = median_height*value, distance = s_distance,plateau_size = 1)
        left = _.get('left_edges')
        iddd = self.get_contig_id(matrix,all_dic)
        idd_list = []
        new_dic = {}
        for i in peaks:
            if mode == None:
                idd = self.find_location(i,all_dic,big_matrix,choro_frame)#get peak contig id
                idd_list.append(idd)
            else:
                idd = self.find_location(i,all_dic,big_matrix,choro_frame)
                contig = all_dic.get(idd)
                length = self.get_length(contig,all_dic,'length')
                if length >= f_value:#filter peaks taht shorter than value
                    idd_list.append(idd)           
        new_dic[iddd] = set(idd_list)
        return new_dic
    

    def get_ids_dict(self,big_mat,info,all_dic,s_distance = 110,mode = None,value = 4,f_value = 500000):
        new_dict = {}
        for i in self.splited_matrix:
            mnn = self.get_ids(i,big_mat,all_dic,info,s_distance,mode,value,f_value)
            new_dict.update(mnn)
        return new_dict
    
    def get_score_column(self,big_mat,dic,all_dic,s_distance = 110,mode = None,value = 4,f_value = 500000):
        info = self.chro_info
        score = 0
        for key,values in dic.items():
            for i in values:
                matrix = all_dic.get(i)
                result = self.get_ids(matrix,big_mat,all_dic,info,s_distance,mode,value,f_value)#will return a dictionary
                #if contig also can detect from contig peak,score + 1
                for d in result.values():
                    if key in d:
                        score = score + 1

        print('peak detection by column',score) 
        
        
    def optimize_result(self,big_mat,dic,all_dic,info,s_distance = 110,mode = None,value = 4,f_value = 500000):
        info = self.chro_info
        nnn_dic = {}
        for key,values in dic.items():
            listt = []
            #same step as column score calculate
            for i in values:
                matrix = all_dic.get(i)
                result = self.get_ids(matrix,big_mat,all_dic,info,s_distance,mode,value,f_value)
                for d in result.values():
                    if key in d:
                        listt.append(i)
            nnn_dic[key] = listt
        return nnn_dic
    
    def get_contig_id(self,matrix,all_dic):
        for key, value in all_dic.items():
            if np.array_equal(value,matrix):
                return key
 
        return "key doesn't exist"
    
    #get contig length or start,end bin index
    def get_length(self,contig,all_dic,dtype = 'bin'):
        fish_chro_info = self.chro_info
        chro_info = self.chro_info
        z = chro_info.iloc[:,0]
        z = z.values.tolist()
        bin_loc = chro_info.iloc[:,2:4] #get start_bin and end_bin dataframe
        start_bin = bin_loc.iloc[:,0]# change to list
        end_bin = bin_loc.iloc[:,1]
        pp = 0
        for i in range(len(z)):
            #find the index of input contig,and get length or start,end bin use that index,directly find from datframe
            if z[i] == str(self.get_contig_id(contig,all_dic)):
                pp = i
        length_list = fish_chro_info.iloc[:,1].values.tolist()#length list
        if dtype == 'length':
            return length_list[pp]
        elif dtype == 'bin':
            return start_bin[pp],end_bin[pp]
        else:
            return z[pp]
        
    def find_location(self,p_index,all_dic,matrix,choro_frame):
        bin_loc = choro_frame.iloc[:,2:4] #get start_bin and end_bin dataframe
        start_bin = bin_loc.iloc[:,0]# change to list
        end_bin = bin_loc.iloc[:,1]# change to list
        for i in range(len(start_bin)):
            if p_index >= start_bin[i] and p_index < end_bin[i]:#  if input index larger or equel to start bin and less than end bin         
                return self.get_contig_id(matrix[start_bin[i]:end_bin[i],:],all_dic)#extract 2D array betwwen start_bin and end_bin and use dictionary find contig id

    
    
    


# In[ ]:





# In[ ]:




