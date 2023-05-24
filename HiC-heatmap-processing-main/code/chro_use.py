#!/usr/bin/env python
# coding: utf-8

# In[12]:


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
from scipy.signal import find_peaks
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.transform import resize
from skimage import data, img_as_float


# In[1]:


#path: The .cool file path input(if use .mcool file rather than .cool file, need to choose 
#resolution by add "::/resolutions/{resolution value} " after path, resolutions value can only
#choose from {1000,5000,10000,25000,50000,100000,250000,500000,1000000,2500000}, The smaller 
#the value, the more detailed the matrix and the image.）
class heatmap_anylsis:
    
    def __init__(self,path):
        self.path = path
        
#use cooler package generate cooler api  
# will return cooler.api.cooler 
    def readcool(self):
        new_cool = cooler.Cooler(self.path)
        return new_cool
    
## Converting imported .cool files into two-dimensional arrays    
    def getmatrix(self,value = False):
        new_cool = cooler.Cooler(self.path)
        new_matrix = new_cool.matrix(sparse=False, balance=value)[:]
        return new_matrix
    
    
    def generatemap(self,matrix,size = (8,8)):
        plt.figure(figsize=size)
        plt.imshow(matrix ** 0.02, cmap='afmhot_r',norm=colors.PowerNorm(gamma=0.45))

    
## generate heatmap from matrix(two-dimensional arrays)  
    def generatemap2(self,matrix,size = (8,8)):
        plt.figure(figsize=size)
        if matrix.shape[0] < 200:
            matrix_c = matrix.copy()
            matrix_c = resize(matrix_c, (250,matrix.shape[1]))
            plt.imshow(matrix_c, cmap='afmhot_r',norm=colors.PowerNorm(gamma=0.15))
            plt.axis('off')
        #resize unvisible graph
        else:
            plt.imshow(matrix ** 0.02, cmap='afmhot_r',norm=colors.PowerNorm(gamma=0.45))
            plt.axis('off')
    
## Table containing bin related informations(start,end position and nomalization information)        
    def bins_info(self):
        all_info = mapp.HicGenome(self.path)
        bin_info = all_info.bins
        return bin_info
    
## Table of chromosome information. Each row contains the name, length, first and last bin of a chromosome.    
    def chro_info(self):
        table_c = io.load_cool(self.path)
        return table_c[1]
    
## thransfer bin start and end location from dataframe to list  
    def thansfer_bin_to_list(self,list_type='start_list'):
        choro_frame = self.chro_info()
        bin_loc = choro_frame.iloc[:,2:4]
        #extract start and end bin column
        start_bin = bin_loc.iloc[:,0]
        end_bin = bin_loc.iloc[:,1]
        #transfer to list
        start_list = start_bin.values.tolist()
        end_list = end_bin.values.tolist()
        if list_type == 'start_list':
            return start_list
        elif list_type == 'end_list':
            return end_list
        else:
            print('Please input "start_list" or "end_list" to choose list type')
            
## split matrix by each contig row(use range of bins i.e. range from start bin to end bin, generate each contig row matrix, also will return a dictionary
## related contig id to matrix
    def split_matrix(self,rtype = 'list'):
        n_list=[]
        n_dic = {}
        chro_info = self.chro_info()
        #extract contig id column and thansfer to list
        z = chro_info.iloc[:,0]
        z = z.values.tolist()
        start_list = self.thansfer_bin_to_list()
        end_list =self.thansfer_bin_to_list('end_list')
        image = self.getmatrix()
        for i in range(len(start_list)):
            #get each contig start bin and end bin position,extract each contig row
            row = image[start_list[i]:end_list[i],:]
            n_dic[z[i]] = row
            n_list.append(row)
        if rtype == 'list':
            return n_list
        elif rtype == 'dic':
            return n_dic
        
## remove array from list       
    def removearray(self,List,arr):
        ind = 0
        size = len(List)
        while ind != size and not np.array_equal(List[ind],arr):
            ind += 1
        if ind != size:
            List.pop(ind)
        else:
            raise ValueError('array not found in list.')
            
    def get_contig_id(self,matrix,my_dict):
        for key, value in my_dict.items():
            if np.array_equal(value,matrix):
                return key
 
        return "key doesn't exist"

## get contig id from matrix that interested
    def sort_matrix(self):
        matrix_list = self.split_matrix()
        m_list = []
        c_list = matrix_list.copy()
        #put width info into m_list,sort matrix by matrix width
        for i in matrix_list:
            m_list.append(i.shape[0])
            m_list.sort()
        sorted_list = []
        #accoring m_list find 2D matrix
        for i in m_list:
            for j in c_list:
                if j.shape[0] == i:
                    sorted_list.append(j)
                    self.removearray(c_list,j)
                    break    
        return sorted_list
    
    def remove_contig_length(self,length):
        chrom_info = self.chro_info() 
        splited_matrix = self.split_matrix()
        #extract contig length column and thansfer to list
        length_list = chrom_info.iloc[:,1].values.tolist() #length list
        list1 = []
        for i in length_list:
            if i >=length: 
                list1.append(i)#list 1 contain all length in the list wich longer than length value
        list_index = []
        for i in range(len(length_list)):
            for j in list1:
                if length_list[i] == j:
                    list_index.append(i)# get contig length index which longer than length value
        list_new = []
        for i in list_index:
            list_new.append(splited_matrix[i])#according index find contig matrix
        return list_new
    
## show peak diagram for contig matrix(cannot use this function to whole heatmap)
    def peak_dia(self,contig_matrix):
        plt.plot(contig_matrix.T)
        plt.show()
        
    def find_location(self,p_index,dic,matrix,choro_frame):
        bin_loc = choro_frame.iloc[:,2:4] #get start_bin and end_bin dataframe
        start_bin = bin_loc.iloc[:,0]# change to list
        end_bin = bin_loc.iloc[:,1]# change to list
        for i in range(len(start_bin)):
            if p_index >= start_bin[i] and p_index < end_bin[i]:  #  if input index larger or equel to start bin and less than end bin         
                idd = self.get_contig_id(matrix[start_bin[i]:end_bin[i],:],dic)#extract 2D array betwwen start_bin and end_bin and use dictionary find contig id
        return idd
    
    

