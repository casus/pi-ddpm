# image loader for the first round data generation

import os
import numpy as np
from skimage.transform import resize
from skimage.util import view_as_windows
             
# for the widefield microscope reconstruction
# for first round of data: input: packed data in npz ['c_img'],['c_psf'],['w_img'],['w_psf'],['o']
class dataGenerator_imageStack:
    def __init__(self, data_dir, data_list, batch_size):
        
        # input the loaction of the data
        self.data_dir = data_dir
        self.data_list = data_list
        self.batch_size = batch_size
        
    def _rescale(imageStack, MIN=0, MAX=1):
        
        if imageStack[0].max() !=1:
            # print('rescale:', MIN, MAX)
            ImageScale = []

            for stack in range(imageStack.shape[0]):
                temp = imageStack[stack,...]
                tempScale = np.interp(temp, (temp.min(), temp.max()), (MIN, MAX))
                ImageScale.append(tempScale.astype('int'))
        else:
            ImageScale = imageStack
        return np.asarray(ImageScale)
        
    def imageLoader(self):

        while True:
            
            for index, dataset_name in enumerate(self.data_list):
                
                temp_dataset = np.load(self.data_dir + dataset_name) 
    
                c_img, c_psf, w_img = temp_dataset['c_img'],temp_dataset['c_psf'],temp_dataset['w_img']  # unpack数据
                w_psf, objects = temp_dataset['w_psf'],temp_dataset['o']

                L = c_img.shape[0]  
            
                batch_start = 0
                batch_end = self.batch_size

                while batch_start < L:

                    limit = min(batch_end, L)
                    
                    # take data out and rescale
                    c_img_temp, c_psf_temp = (c_img[batch_start:limit]), (c_psf[batch_start:limit])
                    w_img_temp, w_psf_temp = (w_img[batch_start:limit]), (w_psf[batch_start:limit])
                    o_temp = (objects[batch_start:limit])
                    
                    # swap the axes [z,y,x] -> [x, y, z]
                    c_img_temp = np.swapaxes(c_img_temp, 1,3)
                    c_psf_temp = np.swapaxes(c_psf_temp, 1,3)
                    w_img_temp = np.swapaxes(w_img_temp, 1,3)
                    w_psf_temp = np.swapaxes(w_psf_temp, 1,3)
                    o_temp = np.swapaxes(o_temp, 1,3)
                    
                    yield(c_img_temp, w_img_temp)  # ouput only the c and w

                    batch_start += self.batch_size 
                    batch_end += self.batch_size

# read in the new simulated dataset: the image is the middle slice
# test['confocal'],test['widefield'], test['object'], test['confocal_kernel'], test['widefield_kernel']
class dataGenerator_2Dstack:
    def __init__(self, data_dir, data_list, batch_size):
        
        # input the loaction of the data
        self.data_dir = data_dir
        self.data_list = data_list
        self.batch_size = batch_size
        
    def _rescale(self, imageStack, MIN=0, MAX=1):

        if imageStack[0].max() !=1:
            
            ImageScale = []

            for stack in range(imageStack.shape[0]):
                temp = imageStack[stack,...]
                tempScale = np.interp(temp, (temp.min(), temp.max()), (MIN, MAX))
                ImageScale.append(tempScale.astype('float64'))
        else:
            ImageScale = imageStack
        return np.asarray(ImageScale)
        
    def imageLoader(self):

        while True:
            
            for index, dataset_name in enumerate(self.data_list):

                print('load dataset:', dataset_name)
                temp_dataset = np.load(self.data_dir + dataset_name)
                
                c_imgs, w_imgs, obj_imgs = temp_dataset['confocal'],temp_dataset['widefield'], temp_dataset['object'] 
                c_psf, w_psf = temp_dataset['confocal_kernel'], temp_dataset['widefield_kernel']

                L = c_imgs.shape[0]  
            
                batch_start = 0
                batch_end = self.batch_size

                while batch_start < L:

                    limit = min(batch_end, L) 

                    c_img_temp, w_img_temp, o_temp = (c_imgs[batch_start:limit]), (w_imgs[batch_start:limit]),(obj_imgs[batch_start:limit]) 
                    c_psf_temp, w_psf_temp = (c_psf[batch_start:limit]), (w_psf[batch_start:limit])
                    c_psf_temp, w_psf_temp = np.expand_dims(c_psf_temp, axis=3), np.expand_dims(w_psf_temp, axis=3)
                    
                    # rescale into [0, 1]
                    c_img_temp, w_img_temp = self._rescale(c_img_temp, MIN=0, MAX=1), self._rescale(w_img_temp, MIN=0, MAX=1)
                    o_temp = self._rescale(o_temp, MIN=0, MAX=1)
                    
                    # expand dimension for the model
                    c_img_temp = np.expand_dims(c_img_temp, axis=3)
                    o_temp = np.expand_dims(o_temp, axis=3)
                    w_img_temp = np.expand_dims(w_img_temp, axis=3)

                    yield(c_img_temp, w_img_temp, o_temp, c_psf_temp, w_psf_temp) # output the unpacked dataset

                    batch_start += self.batch_size 
                    batch_end += self.batch_size