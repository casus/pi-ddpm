import numpy as np
from scipy.signal import fftconvolve
from pyotf.otf import SheppardPSF
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import cupy as cp
import tifffile as tif
import cv2
import os
from pathlib import Path
from skimage.util.shape import view_as_blocks
from skimage.util import montage
from einops import rearrange
from scipy.ndimage import shift
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def disk_kernel(radius):
    full_size = int(np.ceil(radius * 2))
    if full_size % 2 == 0:
        full_size += 1
    coords = np.indices((full_size, full_size)) - (full_size - 1) // 2
    r = np.sqrt((coords ** 2).sum(0))
    kernel = r < radius
    return kernel / np.sum(kernel)


def lightbased_microscopy_psf(shape, numerical_aperture, refractive_index, excitation_wavelength,
                              emission_wavelength, pinhole_size, voxel_size=-1., zres=0.1):
    psf_params = dict(na=numerical_aperture, ni=refractive_index, wl=excitation_wavelength, size=shape[0],
                      vec_corr="none")
    psf_params_em = dict(na=numerical_aperture, ni=refractive_index, wl=emission_wavelength, size=shape[0],
                         vec_corr="none")

    # Set the Nyquist sampling rate
    nyquist_sampling = psf_params["wl"] / psf_params["na"] / 4

    if voxel_size == -1:
        oversample_factor = 2
        psf_params["res"] = nyquist_sampling / oversample_factor / 2
        psf_params_em["res"] = nyquist_sampling / oversample_factor / 2

        psf_params["zres"] = zres
        psf_params_em["zres"] = zres
        # zres=190e-3,
        # zsize=128,
        print(nyquist_sampling / oversample_factor / 2)


    else:
        psf_params["res"] = voxel_size
        psf_params_em["res"] = voxel_size
        psf_params["zres"] = zres
        psf_params_em["zres"] = zres

    # widefield excitation otf
    psf_exc = SheppardPSF(**psf_params).PSFi
    psf_det = SheppardPSF(**psf_params_em).PSFi
    if pinhole_size == -1:
        psf_det_au1 = 1.
    else:
        airy_unit = 1.22 * psf_params_em["wl"] / psf_params_em["na"] / psf_params_em["res"]
        ph_au = (pinhole_size / 1000)
        kernel = disk_kernel(ph_au * airy_unit / 2)
        psf_det_au1 = fftconvolve(psf_det, kernel[None], "same", axes=(1, 2))
    psf = psf_det_au1 * psf_exc

    return psf


def generate_binary_metaballs_3d(shape=(32, 32, 32), n_balls=4, radius_range=(0.3, 0.8), threshold=0.1):
    # Create a grid of points
    x, y, z = cp.meshgrid(cp.linspace(-1, 1, shape[0]), cp.linspace(-1, 1, shape[1]), cp.linspace(-1, 1, shape[2]))
    points = cp.stack([cp.asarray(x), cp.asarray(y), cp.asarray(z)], axis=-1)

    # Generate random ball positions and weights
    positions = cp.random.uniform(-0.9, 0.9, size=(n_balls, 3))
    weights = cp.random.uniform(radius_range[0], radius_range[1], size=(n_balls,))

    # Compute the distance to each ball
    distances = cp.sqrt(np.sum((points[:, :, :, cp.newaxis] - positions) ** 2, axis=-1))

    # Compute the field value at each point using the weighted sum of the inverse distances
    field = cp.sum(weights / distances, axis=-1)
    field /= cp.amax(field)

    # Threshold the field to generate the isosurface
    # binary = field > threshold
    binary = cp.zeros(shape)
    binary[field > threshold] = 1
    binary_cpu = binary.get()  # transfer data back to host

    return binary_cpu


def add_low_frequency_noise(img_gray, freq_range):
    # Generate random frequencies for the noise
    freq_x = np.random.uniform(freq_range[0], freq_range[1])
    freq_y = np.random.uniform(freq_range[0], freq_range[1])
    freq_z = np.random.uniform(freq_range[0], freq_range[1])

    # Generate low-frequency noise using sine waves
    x, y, z = np.meshgrid(np.linspace(-1, 1, img_gray.shape[0]),
                          np.linspace(-1, 1, img_gray.shape[1]),
                          np.linspace(-1, 1, img_gray.shape[2]))
    noise = np.sin(2 * np.pi * freq_x * x +
                   2 * np.pi * freq_y * y +
                   2 * np.pi * freq_z * z)

    # Scale the noise to the range [0, 1]
    noise -= noise.min()
    noise /= noise.max()

    # Add the noise to the grayscale image
    img_gray = np.clip(img_gray + noise, 0, 1)

    return img_gray


def generate_synthetic_sample(shape_img, shape_kernel, numerical_aperture, refractive_index, excitation_wavelength,
                              emission_wavelength, pinhole_size, voxel_size=-1., zres=0.1, n_balls=50, threshold=0.05,
                              freq_range=(0.1, 0.5), focal_plane_shift=0, projection_type='avg',
                              dataset_type='metaballs', im_net_paths=None):
    # generate psf according to user params

    psf = lightbased_microscopy_psf(shape_kernel, numerical_aperture, refractive_index,
                                    excitation_wavelength, emission_wavelength, pinhole_size,
                                    voxel_size=voxel_size, zres=zres)
    psf_w = lightbased_microscopy_psf(shape_kernel, numerical_aperture, refractive_index,
                                      excitation_wavelength, emission_wavelength, -1,
                                      voxel_size=voxel_size, zres=zres)
    # psf = np.transpose(psf, (2, 1, 0))

    # generate synthetic sample
    if dataset_type == 'metaballs':
        img_bin = generate_binary_metaballs_3d(shape_img, n_balls=n_balls, threshold=threshold)
        img_gray = distance_transform_edt(1 - img_bin)
        img_gray = add_low_frequency_noise(img_gray, freq_range=freq_range)
        img_gray[img_gray == 1.0] = 0
    else:
        # load imgs from inet.

        ix = np.random.randint(0, len(im_net_paths))
        path = str(im_net_paths[ix])
        img = cv2.imread(path)
        img = cv2.resize(img, (shape_img[1], shape_img[0]), interpolation=cv2.INTER_AREA)
        img_gray = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=0) / 255.

    psf /= np.max(psf)
    psf_w /= np.max(psf_w)

    if projection_type == 'max':
        p_img = np.max(img_gray, axis=0)
        psf = shift(psf, (focal_plane_shift, 0, 0))
        psf = np.max(psf, axis=0)
        psf_w = shift(psf_w, (focal_plane_shift, 0, 0))
        psf_w = np.max(psf_w, axis=0)
        syn_sample = fftconvolve(psf, p_img, 'same')
        syn_sample_w = fftconvolve(psf_w, p_img, 'same')

    elif projection_type == 'min':
        p_img = np.min(img_gray, axis=0)
        psf = shift(psf, (focal_plane_shift, 0, 0))
        psf = np.min(psf, axis=0)
        psf_w = shift(psf_w, (focal_plane_shift, 0, 0))
        psf_w = np.min(psf_w, axis=0)
        syn_sample = fftconvolve(psf, p_img, 'same')
        syn_sample_w = fftconvolve(psf_w, p_img, 'same')

    elif projection_type == 'avg':
        p_img = np.mean(img_gray, axis=0)

        psf = shift(psf, (focal_plane_shift, 0, 0))
        psf = np.mean(psf, axis=0)
        psf_w = shift(psf_w, (focal_plane_shift, 0, 0))
        psf_w = np.mean(psf_w, axis=0)
        syn_sample = fftconvolve(psf, p_img, 'same')
        syn_sample_w = fftconvolve(psf_w, p_img, 'same')

    # syn_sample = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(ft_sample * otf)))

    return syn_sample, syn_sample_w, p_img, psf, psf_w


def generate_training_set(n_samples, n_batches, sample_shape, kernel_shape, paths=None):
    CTrain = np.zeros((n_samples,) + sample_shape[:-1])
    WTrain = np.zeros((n_samples,) + sample_shape[:-1])
    YTrain = np.zeros((n_samples,) + sample_shape[:-1])
    CKTrain = np.zeros((n_samples,) + sample_shape[:-1])
    WKTrain = np.zeros((n_samples,) + sample_shape[:-1])

    for b_idx in tqdm(range(n_batches)):
        count = 0
        while count < n_samples:
            try:
                numerical_aperture = np.random.uniform(0.4, 1.0)
                excitation_wavelength = np.random.uniform(0.32, 0.4)
                emission_wavelength = np.random.uniform(0.45, 0.55)
                pinhole_size = np.random.uniform(0.1, 1000)
                focal_plane_shift = 0
                refractive_index = 1.33
                resolution = np.random.uniform(0.04 * kernel_shape[0] / 64,
                                               0.06 * kernel_shape[0] / 64)  # 64 sized kernel constants
                zres = np.random.uniform(0.055 * kernel_shape[0] / 64,
                                         0.068 * kernel_shape[0] / 64)  # 64 sized kernel constants

                threshold = np.random.uniform(0.34, 0.36)
                n_balls = np.random.randint(100, 200)

                # Example usage
                syn_sample, syn_sample_w, img_gray, psf, psf_w = generate_synthetic_sample(sample_shape, kernel_shape,
                                                                                           numerical_aperture,
                                                                                           refractive_index,
                                                                                           excitation_wavelength,
                                                                                           emission_wavelength,
                                                                                           pinhole_size, resolution,
                                                                                           zres, n_balls=n_balls,
                                                                                           threshold=threshold,
                                                                                           focal_plane_shift=focal_plane_shift,
                                                                                           dataset_type='inet',
                                                                                           im_net_paths=paths)

                CTrain[count] = syn_sample
                WTrain[count] = syn_sample_w
                YTrain[count] = img_gray
                CKTrain[count] = psf
                WKTrain[count] = psf_w

                count += 1
                print(count, end='\r')
            except Exception as e:
                print(e)
                continue
        print('saving data')
        np.savez('./data/microscope_ds_imnet_test' + str(b_idx) + '.npz', confocal=CTrain,
                 widefield=WTrain,
                 confocal_kernel=CKTrain, widefield_kernel=WKTrain, object=YTrain)
        print('saved data')


def norm_01(x):
    if (np.amax(x) - np.amin(x)) != 0:
        n_x = (x - np.amin(x, axis=(1, 2, 3), keepdims=True)) / (
                np.amax(x, axis=(1, 2, 3), keepdims=True) - np.amin(x, axis=(1, 2, 3), keepdims=True))
    else:
        n_x = 0
    return n_x


def generate_biosr_dataset(base_path):
    parent_path_blurry = base_path + '/training_upscaled_wf_dataset/'
    paths_blurry = list(Path(parent_path_blurry).glob('**/*.tif'))
    XTrain = np.zeros((len(paths_blurry), 256, 256, 1))
    YTrain = np.zeros((len(paths_blurry), 256, 256, 1))
    for idx in tqdm(range(len(paths_blurry))):
        blurry_img_path = paths_blurry[idx]
        img_path_blurry = str(blurry_img_path)
        img_path_gt = img_path_blurry.replace('training_upscaled_wf_dataset', 'training_gt_dataset').replace(
            'wf_upscaled', 'gt')

        im_blurry = tif.imread(img_path_blurry)
        im_gt = tif.imread(img_path_gt)
        if im_blurry.shape[0] != 256 or im_gt.shape[0] != 256:
            im_blurry = cv2.resize(im_blurry, (256, 256))
            im_gt = cv2.resize(im_gt, (256, 256))

        XTrain[idx, :, :, 0] = norm_01(im_blurry) * 2 - 1
        YTrain[idx, :, :, 0] = norm_01(im_gt) * 2 - 1
    np.savez('./data/biosr_ds.npz', x=XTrain, y=YTrain)


def generate_w2s_dataset(base_path, p_size=256):
    parent_path_blurry = base_path + '/raw/'
    subfolders = [f.path for f in os.scandir(parent_path_blurry) if f.is_dir()]
    XTrain = []
    YTrain = []
    for idx in tqdm(range(30)):
        subfolder = subfolders[idx]
        sim_0 = np.expand_dims(rearrange(view_as_blocks(np.load(f'{subfolder}/sim_channel0.npy'), (p_size, p_size)),
                                         'p q h w -> (p q) h w'), -1)
        sim_1 = np.expand_dims(rearrange(view_as_blocks(np.load(f'{subfolder}/sim_channel1.npy'), (p_size, p_size)),
                                         'p q h w -> (p q) h w'), -1)
        sim_2 = np.expand_dims(rearrange(view_as_blocks(np.load(f'{subfolder}/sim_channel2.npy'), (p_size, p_size)),
                                         'p q h w -> (p q) h w'), -1)

        wf_0 = np.expand_dims(
            rearrange(view_as_blocks(np.load(f'{subfolder}/wf_channel0.npy'), (400, p_size // 2, p_size // 2)),
                      't p q r h w -> (t p q) r  h w'), -1)
        wf_1 = np.expand_dims(
            rearrange(view_as_blocks(np.load(f'{subfolder}/wf_channel1.npy'), (400, p_size // 2, p_size // 2)),
                      't p q r h w -> (t p q) r h w'), -1)
        wf_2 = np.expand_dims(
            rearrange(view_as_blocks(np.load(f'{subfolder}/wf_channel2.npy'), (400, p_size // 2, p_size // 2)),
                      't p q r h w -> (t p q) r h w'), -1)


        YTrain.append(sim_0)
        YTrain.append(sim_1)
        YTrain.append(sim_2)
        XTrain.append(wf_0)
        XTrain.append(wf_1)
        XTrain.append(wf_2)

    XTrain = np.array(XTrain)
    YTrain = np.array(YTrain)

    np.savez('/media/gabriel/data_hdd/w2s_test.npz', x=XTrain, y=YTrain)


#generate_w2s_dataset('/media/gabriel/data_hdd/W2S_raw')
