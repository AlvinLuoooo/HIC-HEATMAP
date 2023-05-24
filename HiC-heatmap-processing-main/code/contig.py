#!/usr/bin/env python
# coding: utf-8

# In[3]:


import chro_use as cu
import column as co
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


# In[2]:


import column as co


# In[1]:


import chro_use as cu


# In[ ]:





# In[2]:


class contig:
    def __init__(self,chro_info,splitted_matrix):
        self.chro_info = chro_info
        self.splitted_matrix = splitted_matrix
        
        
    def get_contig_id(self,matrix,my_dict):
        for key, value in my_dict.items():
            if np.array_equal(value,matrix):
                return key
 
        return "key doesn't exist"
        
    def get_length(self,contig,dic,dtype = 'bin'):
        chro_info = self.chro_info
        z = chro_info.iloc[:,0]
        z = z.values.tolist()
        bin_loc = chro_info.iloc[:,2:4] #get start_bin and end_bin dataframe
        start_bin = bin_loc.iloc[:,0]# change to list
        end_bin = bin_loc.iloc[:,1]
        pp = 0
        for i in range(len(z)):
            if z[i] == str(self.get_contig_id(contig,dic)):
                pp = i
        length_list = chro_info.iloc[:,1].values.tolist()#length list
        if dtype == 'length':
            return length_list[pp]
        elif dtype == 'bin':
            return start_bin[pp],end_bin[pp]
        else:
            return z[pp]
        
    def find_ids_contig(self,all_dic,matrix,mode = None,value = 4,f_value = 500000):
        z = self.chro_info.iloc[:,0]
        bin_loc = self.chro_info.iloc[:,2:4]
        start_bin = bin_loc.iloc[:,0]
        end_bin = bin_loc.iloc[:,1]
        start = start_bin.values.tolist()
        end = end_bin.values.tolist()
        new_dict = {}
        iddd = self.get_contig_id(matrix,all_dic)
        dd = []
        for j in range(len(start)):
            #avoid dd too long
            if len(dd) >= len(start):
                dd.clear
            if mode == None:
                matrixs = matrix[: , start[j]:end[j]]
                dd.append(matrixs.mean())
            elif mode == 'smooth':
                #gaussian filter for every conytig matrix in a contig row
                matrixs = ndi.filters.gaussian_filter(matrix[: , start[j]:end[j]],(3,3))
                dd.append(matrixs.mean())
            else:
                matrixs = matrix[: , start[j]:end[j]]
                dd.append(matrixs.mean())
        ipo = []
        mean = np.mean(dd)
        peaks, _ = find_peaks(dd,height = mean * value)#result will be the index of contig
        for i in peaks:
            if mode == 'filter':
                #get contig matrix accoring index
                matrix = all_dic.get(z[i])
                length = self.get_length(matrix,all_dic,dtype = 'length')
                if length >= f_value:
                    ipo.append(z[i])
            else:
                ipo.append(z[i])
        new_dict[iddd] = set(ipo)
        return new_dict
    
    #calculate contig score
    def get_score_contig(self,dic,all_dic,mode = None,value = 4,f_value = 500000):
        score = 0
        for key,values in dic.items():
            for i in values:
                matrix = all_dic.get(i)
                result = self.find_ids_contig(all_dic,matrix,mode,value,f_value)
                for d in result.values():
                    if key in d:
                        score = score + 1
        print('peak detection by contig',score) 
        
    def get_score_contig_af(self,dic):
        score = 0
        for key,values in dic.items():
            for i in values:
                result = dic.get(i)
                if key in result:
                    score = score + 1
        print('peak detection by contig after filter',score)
        
    def sort_by_contig(self,all_dic,mode = None, value = 4,f_value = 500000):
        new_dict = {}
        for i in self.splitted_matrix:
            ll = self.find_ids_contig(all_dic,i,mode,value,f_value)
            new_dict.update(ll)
        return new_dict
    
    def optimize_result(self,dic,all_dic,mode = None,value = 4,f_value = 500000):
        info = self.chro_info
        nnn_dic = {}
        for key,values in dic.items():
            listt = []
            for i in values:
                matrix = all_dic.get(i)
                result = self.find_ids_contig(all_dic,matrix,mode,value,f_value)                
                for d in result.values():
                    if key in d:
                        listt.append(i)
            nnn_dic[key] = listt
        return nnn_dic
        

        
    
    

