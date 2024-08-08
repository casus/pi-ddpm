from scripts.data_scripts.dataset_generation import norm_01
import numpy as np
import matplotlib.pyplot as plt
from src.models.pi_ddpm import train_model
from tqdm import tqdm
from src.training_utils.ddpm_utils import variance_schedule
import tensorflow as tf
from phantominator import shepp_logan
from src.training_utils.forward_models import forward_model_conv
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def norm_2(x):
    return np.sqrt(np.sum(x ** 2, axis=(1, 2, 3), keepdims=True))


def msssim(img1, img2):
    im1 = np.expand_dims(np.expand_dims(img1, -1), 0)
    im2 = np.expand_dims(np.expand_dims(img2, -1), 0)
    return tf.image.ssim_multiscale(im1, im2, max_val=1, filter_size=5, filter_sigma=0.75).numpy()[0]


def ddpm_obtain_sr_img(fx, timesteps_test, p_model, smooth_factor=0.01, reg_type='l1', nu_lr=0.01):
    pred_sr = np.random.normal(0, 1, fx.shape)
    gamma_vec_test, alpha_vec_test = variance_schedule(timesteps_test, schedule_type='linear', min_beta=1e-4,
                                                       max_beta=3e-2)

    for t in tqdm(range(timesteps_test, 0, -1)):
        z = np.random.normal(0, 1, fx.shape)
        if t == 1:
            z = 0
        alpha_t = alpha_vec_test[t - 1]
        gamma_t = gamma_vec_test[t - 1]
        gamma_tm1 = gamma_vec_test[t - 2]
        beta_t = 1 - alpha_t

        gamma_t_inp = np.ones((fx.shape[0], 1)) * np.reshape(gamma_t, (1, 1))
        beta_factor = (1 - gamma_tm1) * beta_t / (1 - gamma_t)
        sigma = beta_factor
        pred_params = p_model.predict([pred_sr, fx, gamma_t_inp], verbose=False)
        if pred_params.shape[-1] == 2:
            pred_noise = pred_params[..., 0:1]
            pred_grad = pred_params[..., 1:]
            pred_grad = pred_grad * nu_lr * np.sqrt(gamma_t) / norm_2(pred_grad)
        else:
            pred_noise = pred_params
            pred_grad = np.zeros_like(pred_params)

        alpha_factor = beta_t / np.sqrt(1 - gamma_t)
        # you can change the variance from beta factor to beta_t to see the difference between the different beta schedules.

        if reg_type == 'l1':
            pred_sr = 1 / np.sqrt(alpha_t) * (pred_sr - alpha_factor * pred_noise) + np.sqrt(
                sigma) * z - smooth_factor * np.sign(pred_sr) * np.sqrt(sigma) - pred_grad * np.sqrt(sigma)

        if reg_type == 'l2':
            pred_sr = 1 / np.sqrt(alpha_t) * (pred_sr - alpha_factor * pred_noise) + np.sqrt(
                sigma) * z - smooth_factor * np.sqrt(sigma) * (2 * pred_sr) - pred_grad * np.sqrt(sigma)


    return pred_sr


def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x - np.percentile(x, min_prc, axis=(1, 2, 3), keepdims=True)) / (
            np.percentile(x, max_prc, axis=(1, 2, 3), keepdims=True) - np.percentile(x, min_prc,
                                                                                     axis=(1, 2, 3),
                                                                                     keepdims=True) + 1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


def test_sr_ddpm(model_version, test='shepp-logan'):
    timesteps_test = 200
    patch_size = 128
    img_channels = 1
    p_model_pinn, t_model_pinn = train_model((patch_size, patch_size, 1), (patch_size, patch_size, img_channels),
                                             img_channels,
                                             noise_est=False)
    t_model_pinn.load_weights('./models_weights/pinn_finetunned/model' + str(model_version) + '.h5')

    # point this path to the testing data in your folder
    if test == 'shepp-logan':
        phantom_stack, _, _ = shepp_logan((128, 128, 20), MR=True)
        data_blobs = np.load('./data/microscope_ds_nx_0.npz')
        KTest = np.expand_dims(data_blobs['widefield_kernel'], -1)
        kx = np.random.randint(0, KTest.shape[0], 2)
        fk = KTest[kx]
        fy = norm_01(np.expand_dims(phantom_stack[..., 7:8], 0)) * 2 - 1

        fx = norm_01(tf.abs(forward_model_conv([fk[1:2], fy * 0.5 + 0.5]))) * 2 - 1
        lam = np.random.uniform(0.005, 0.005)
        n_fx = norm_01(fx) + np.random.normal(lam, lam, fx.shape)
        n_fx = norm_01(n_fx) * 2 - 1

        pred_fy_pinn = ddpm_obtain_sr_img(n_fx, timesteps_test, p_model_pinn, smooth_factor=0.015, reg_type='l1')
        plt.imsave('./teaser_wf.png', np.squeeze(prctile_norm(pred_fy_pinn + n_fx, min_prc=12)), cmap='gray')


    elif test == 'widefield':
        data = np.load('./data/teaser_c_w_test.npz')
        XTest = np.expand_dims(data['w_img'][..., 19:20], 0)
        YTest = np.expand_dims(data['c_img'][..., 19:20], 0)
        objects = np.zeros_like(XTest)
        reg = 'l1'
        r_weight = 0.015
        e_weight = 1
        for idx in range(XTest.shape[0]):
            fx = XTest[idx:idx + 1]
            fy = YTest[idx:idx + 1]
            if np.amax(fy) >= 0.2:
                print(np.amax(fy), np.amin(fy), idx)
                fx = norm_01(fx) * 2 - 1
                fy = norm_01(fy) * 2 - 1
                t = time.time()
                pred_dfy_pinn = ddpm_obtain_sr_img(fx, timesteps_test, p_model_pinn, smooth_factor=r_weight, nu_lr=e_weight, reg_type=reg)
                objects[idx, :, :, 0] = pred_dfy_pinn[0, :, :, 0] + fx[0, :, :, 0]
                plt.imsave('./teaser_wf.png', np.squeeze(prctile_norm(pred_dfy_pinn + fx, min_prc=12)), cmap='gray')



    elif test == 'confocal':
        data = np.load('./teaser_c_w_test.npz')

        XTest = np.expand_dims(data['c_img'], 0)
        YTest = np.expand_dims(data['c_img'], 0)
        objects = np.zeros_like(XTest)
        for idx in range(0, XTest.shape[0]):
            for slice_idx in range(19, 20):
                fx = norm_01(XTest[idx:idx + 1, :, :, slice_idx:slice_idx + 1]) * 2 - 1
                fy = norm_01(YTest[idx:idx + 1, :, :, slice_idx:slice_idx + 1]) * 2 - 1
                if np.amax(norm_01(YTest[..., 19:20])[idx:idx + 1] * 2 - 1) >= 0:
                    pred_fy = ddpm_obtain_sr_img(fx, timesteps_test, p_model_pinn, smooth_factor=0.015, nu_lr=1.,
                                                 reg_type='l1')
                    objects[idx, :, :, slice_idx] = pred_fy[0, :, :, 0] + fx[0, :, :, 0]
                    plt.imsave(f'./teaser_confocal_{slice_idx}.png',
                               np.squeeze(prctile_norm(pred_fy + fx, min_prc=12)), cmap='gray')



test_sr_ddpm(225000, test='widefield')
# test_unet_biosr()
