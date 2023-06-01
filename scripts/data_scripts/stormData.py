import os

import glob
import tifffile
import numpy as np
from skimage.transform import resize
import cv2

# rescale the image stack [t, W,H,ch]
def rescaleStack(imageStack, MIN, MAX):
    
    ImageScale = []
    for stack in range(imageStack.shape[0]):
        temp = imageStack[stack,...]
        tempScale = np.interp(temp, (temp.min(), temp.max()), (MIN, MAX))
        # print(stack, tempScale.min(), tempScale.max())
        ImageScale.append(tempScale.astype('float64'))
    return np.asarray(ImageScale)

def shiftDetect(REF, IMG):

    image1, image2 = REF, IMG
    gray1 = cv2.normalize(cv2.cvtColor((image1 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    gray2 = cv2.normalize(cv2.cvtColor((image2 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)

    # registration
    shift, _ = cv2.phaseCorrelate(gray1, gray2)

    shift_x, shift_y = shift
    shift = np.asarray(shift)
    reg_img = np.roll(image2, np.round(shift).astype('int'), axis=(0, 1))
    print('shifts:', np.round(shift).astype('int'))
    
    return shift, reg_img

def patchStacks(IMG, SIZE):
    
    image, patch_size = IMG, SIZE

    num_patches = image.shape[0] // patch_size
    SHAPE = image.shape

    patches = np.empty((num_patches, num_patches, patch_size, patch_size, 3))
    # Loop through the image and extract patches
    for i in range(num_patches):
        for j in range(num_patches):
            patches[i, j] = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
            
    return patches, np.asarray(SHAPE), num_patches

def pre_STORM_main(PATH, SAVE_PATH, P_SIZE=128):
    # INPUT: original path; save path; p_size

    data_list = glob.glob(os.path.join(PATH, "*.tif"))  # ensure img and msk paired
    print(data_list)

    # read in data
    w_storm = tifffile.imread(data_list[0])
    w_storm = np.expand_dims(w_storm, axis=3)
    w_storm = np.swapaxes(w_storm, 0, 3)
    w_storm = rescaleStack(w_storm, 0, 1)

    storm = tifffile.imread(data_list[1])
    storm = np.expand_dims(storm, axis=3)
    storm = np.swapaxes(storm, 0, 3)

    # Load the uint16 image stack
    image_stack_uint16 = storm[0]  
    image_stack_uint8 = image_stack_uint16.astype(np.uint8)  

    # Resize 
    storm_pre = resize(image_stack_uint8, (512, 512, 3), anti_aliasing=True)
    storm_pre = rescaleStack(np.expand_dims(storm_pre, axis=0), 0, 1)
    storm_pre = storm_pre.squeeze()

    # register
    w_storm_pre = w_storm[0]
    shifts, w_storm_pre_reg = shiftDetect(storm_pre, w_storm_pre)

    # patchify
    w_patches, w_image_shape, w_num_patches = patchStacks(w_storm_pre_reg, P_SIZE)
    patches, image_shape, num_patches = patchStacks(storm_pre, P_SIZE)

    w_flat_patches = np.reshape(w_patches, (4*4, P_SIZE, P_SIZE, 3))
    gt_flat_patches = np.reshape(patches, (4*4, P_SIZE, P_SIZE, 3))

    # save the file
    np.savez(SAVE_PATH + 'wP_stormP_oriW_oriStorm_unpatchify.npz', w_patches=w_patches, s_patches=patches, w_ori=w_storm_pre_reg, storm=storm_pre,
            shape = image_shape, num = num_patches)
    

# example
pre_STORM_main(PATH='/bigdata/casus/MLID/RuiLi/Data/LM/deepNuclei/test/STORM/', 
               SAVE_PATH='/bigdata/casus/MLID/RuiLi/Data/LM/deepNuclei/test/STORM/')


