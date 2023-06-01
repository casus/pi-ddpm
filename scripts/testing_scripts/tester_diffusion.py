from scripts.data_scripts.dataset_generation import norm_01
import numpy as np
import matplotlib.pyplot as plt
from src.models.pi_ddpm import train_model
from tqdm import tqdm
from src.training_utils.ddpm_utils import variance_schedule
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
from keras.layers import *
from keras.models import Model

from phantominator import shepp_logan
from src.training_utils.forward_models import forward_model_conv
from pathlib import Path


def tv_img_grad(x):
    tv_img = tf.image.total_variation(x)
    grads = tf.gradients(tv_img, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)[0]
    return grads


def tv_reg_grad(pred_sr):
    input_img = Input(pred_sr.shape[1:])
    tv_value_grad = Lambda(tv_img_grad)(input_img)
    model = Model(input_img, tv_value_grad)
    model.compile(loss='mae')
    return model


def ddpm_obtain_sr_img(fx, timesteps_test, p_model, inf_type='lang', smooth_factor=0.01, reg_type='l1',
                       save_inter_steps=False):
    pred_sr = np.random.normal(0, 1, fx.shape)
    gamma_vec_test, alpha_vec_test = variance_schedule(timesteps_test, schedule_type='linear', min_beta=1e-4,
                                                       max_beta=3e-2)
    tv_reg = tv_reg_grad(pred_sr)
    print(pred_sr.shape)
    if save_inter_steps:
        inter_steps = []

    # smooth_factor = 0.01
    for t in tqdm(range(timesteps_test, 0, -1)):
        z = np.random.normal(0, 1, fx.shape)
        if t == 1:
            z = 0
        alpha_t = alpha_vec_test[t - 1]
        gamma_t = gamma_vec_test[t - 1]
        gamma_tm1 = gamma_vec_test[t - 2]
        beta_t = 1 - alpha_t
        gamma_t_inp = np.ones((fx.shape[0], 1)) * np.reshape(gamma_t, (1, 1))
        pred_noise = p_model.predict([pred_sr, fx, gamma_t_inp], verbose=False)
        if inf_type == 'lang':
            # pred_noise = (pred_sr - np.sqrt(gamma_t) * x_0) / np.sqrt(1 - gamma_t)

            alpha_factor = beta_t / np.sqrt(1 - gamma_t)
            # you can change the variance from beta factor to beta_t to see the difference between the different beta schedules.
            beta_factor = (1 - gamma_tm1) * beta_t / (1 - gamma_t)
            if reg_type == 'l1':
                pred_sr = 1 / np.sqrt(alpha_t) * (pred_sr - alpha_factor * pred_noise) + np.sqrt(
                    beta_t) * z - smooth_factor * np.sign(pred_sr) * np.sqrt(beta_t)
            if reg_type == 'tv':
                pred_sr = 1 / np.sqrt(alpha_t) * (pred_sr - alpha_factor * pred_noise) + np.sqrt(
                    beta_t) * z - np.sqrt(beta_t) * (
                                  smooth_factor * tv_reg.predict(pred_sr, verbose=False) + 0.015 * np.sign(pred_sr))
            if reg_type == 'l2':
                pred_sr = 1 / np.sqrt(alpha_t) * (pred_sr - alpha_factor * pred_noise) + np.sqrt(
                    beta_t) * z - smooth_factor * np.sqrt(beta_t) * (2 * pred_sr)
            if reg_type == 'l2l1':
                pred_sr = 1 / np.sqrt(alpha_t) * (pred_sr - alpha_factor * pred_noise) + np.sqrt(
                    beta_t) * z - np.sqrt(beta_t) * (smooth_factor * (2 * pred_sr) + 0.015 * np.sign(pred_sr))



        elif inf_type == 'ddim':
            pred_x0 = (pred_sr - np.sqrt(1 - gamma_t) * pred_noise) / np.sqrt(gamma_t)
            pred_sr = np.sqrt(gamma_tm1) * pred_x0 + np.sqrt(1 - gamma_tm1) * pred_noise

        if save_inter_steps:
            inter_steps.append(pred_sr + fx)

    if save_inter_steps:
        return pred_sr, inter_steps
    else:
        return pred_sr


def plot_imgs(generated_imgs_pinn, generated_imgs_vanilla, gt, fx, experiment='widefield', model_type='vanilla',
              step=-1, slc=-1, inter_steps=None):
    print(generated_imgs_pinn.shape)
    ssim_pinn = []
    ssim_vanilla = []
    mae_pinn = []
    mae_vanilla = []
    Path('./imgs_output/' + model_type + '_testing/' + experiment + '/').mkdir(parents=True, exist_ok=True)
    f = open('./imgs_output/' + model_type + '_testing/' + experiment + '/exp_summary.txt', "a")
    np.savez('./imgs_output/' + model_type + '_testing/' + experiment + 'raw_results.npz',
             pinn_rec=generated_imgs_pinn + fx, vanilla_rec=generated_imgs_vanilla + fx, gt=gt, input=fx)
    for idx in range(generated_imgs_pinn.shape[0]):
        if step == -1:
            step = idx
        if slc == -1:
            slc = 0

        int_pfy = np.squeeze(norm_01(generated_imgs_pinn[idx:idx + 1, :, :, 0:1] + fx[idx:idx + 1, :, :, 0:1]))
        int_pfy_v = np.squeeze(norm_01(generated_imgs_vanilla[idx:idx + 1, :, :, 0:1] + fx[idx:idx + 1, :, :, 0:1]))
        int_gt = np.squeeze(norm_01(gt[idx:idx + 1, :, :, 0:1]))
        int_x = np.squeeze(norm_01(fx[idx:idx + 1, :, :, 0:1]))

        print(ssim(np.float32(int_pfy), np.float32(int_gt), data_range=1))
        ssim_pinn.append(ssim(np.float32(int_pfy), np.float32(int_gt), data_range=1))

        print(ssim(np.float32(int_pfy_v), np.float32(int_gt), data_range=1))
        ssim_vanilla.append(ssim(np.float32(int_pfy_v), np.float32(int_gt), data_range=1))
        print(np.mean(np.abs(int_pfy - int_gt)))
        mae_pinn.append(np.mean(np.abs(int_pfy - int_gt)))
        print(np.mean(np.abs(int_pfy_v - int_gt)))
        mae_vanilla.append(np.mean(np.abs(int_pfy_v - int_gt)))
        f.write('img_sr' + str(idx) + '_' + str(slc) + ' ssim pinn ' + str(
            ssim(np.float32(int_pfy), np.float32(int_gt), data_range=1)) + '\n')
        f.write('img_sr' + str(idx) + '_' + str(slc) + ' ssim vanilla ' + str(
            ssim(np.float32(int_pfy_v), np.float32(int_gt), data_range=1)) + '\n')
        f.write('img_sr' + str(idx) + '_' + str(slc) + ' mae pinn ' + str(np.mean(np.abs(int_pfy - int_gt))) + '\n')
        f.write(
            'img_sr' + str(idx) + '_' + str(slc) + ' mae vanilla ' + str(np.mean(np.abs(int_pfy_v - int_gt))) + '\n')

        plt.imsave(
            './imgs_output/' + model_type + '_testing/' + experiment + '/img_pinn' + str(idx) + '_' + str(slc) + '.svg',
            np.uint8(np.clip(int_pfy, -1, 1) * 127.5 + 127.5),
            cmap='gray')
        plt.imsave(
            './imgs_output/' + model_type + '_testing/' + experiment + '/img_vanilla' + str(idx) + '_' + str(slc) + '.svg',
            np.uint8(np.clip(int_pfy_v, -1, 1) * 127.5 + 127.5),
            cmap='gray')
        plt.imsave(
            './imgs_output/' + model_type + '_testing/' + experiment + '/img_gt' + str(idx) + '_' + str(slc) + '.svg',
            np.uint8(np.clip(int_gt, -1, 1) * 127.5 + 127.5),
            cmap='gray')
        plt.imsave(
            './imgs_output/' + model_type + '_testing/' + experiment + '/img_wf' + str(idx) + '_' + str(slc) + '.svg',
            np.uint8(np.clip(int_x, -1, 1) * 127.5 + 127.5),
            cmap='gray')
        if inter_steps is not None:
            Path('./imgs_output/' + model_type + '_testing/' + experiment + '/img' + str(idx) + '_' + str(slc)).mkdir(
                parents=True, exist_ok=True)
            for s_idx, c_state in enumerate(inter_steps):
                plt.imsave(
                    './imgs_output/' + model_type + '_testing/' + experiment + '/img' + str(idx) + '_' + str(
                        slc) + '/img_step' + str(s_idx) + '.svg',
                    np.uint8(np.squeeze(norm_01(c_state[idx:idx + 1, :, :, 0:1]) * 2 - 1) * 127.5 + 127.5),
                    cmap='gray')

    f.write('mean ssim pinn ' + str(np.mean(ssim_pinn)) + '\n')
    f.write('mean ssim vanilla ' + str(np.mean(ssim_vanilla)) + '\n')
    f.write('mean mae pinn ' + str(np.mean(mae_pinn)) + '\n')
    f.write('mean mae vanilla ' + str(np.mean(mae_vanilla)) + '\n')
    f.close()

    return ssim_pinn, ssim_vanilla, mae_pinn, mae_vanilla


def test_sr_ddpm(model_version, test='fish'):
    timesteps_test = 600
    patch_size = 128
    img_channels = 1
    p_model_vanilla, t_model_vanilla = train_model((patch_size, patch_size, 1), (patch_size, patch_size, img_channels),
                                                   img_channels,
                                                   noise_est=True)
    p_model_pinn, t_model_pinn = train_model((patch_size, patch_size, 1), (patch_size, patch_size, img_channels),
                                             img_channels,
                                             noise_est=False)
    model_type = 'pinn'
    # this will also load the weights of p_model
    t_model_vanilla.load_weights('./models_weights/vanilla/model' + str(model_version) + '.h5')
    t_model_pinn.load_weights('./models_weights/pinn/model' + str(model_version) + '.h5')

    # point this path to the testing data in your folder
    if test == 'shepp-logan':
        phantom_stack, _, _ = shepp_logan((128, 128, 20), MR=True)
        data_blobs = np.load('./data/microscope_ds_nx_0.npz')
        KTest = np.expand_dims(data_blobs['widefield_kernel'], -1)
        kx = np.random.randint(0, KTest.shape[0], 2)
        fk = KTest[kx]
        fy = norm_01(np.expand_dims(phantom_stack[..., 7:8], 0)) * 2 - 1
        # lena_img = cv2.resize(cv2.cvtColor(cv2.imread('voldie.png'), cv2.COLOR_BGR2GRAY), (patch_size, patch_size),
        #                      interpolation=cv2.INTER_AREA)
        # fy = norm_01(np.expand_dims(np.expand_dims(lena_img, 0), -1)) * 2 - 1

        fx = norm_01(tf.abs(forward_model_conv([fk[1:2], fy * 0.5 + 0.5]))) * 2 - 1
        lam = np.random.uniform(0.005, 0.005)
        n_fx = norm_01(fx) + np.random.normal(lam, lam, fx.shape)
        n_fx = norm_01(n_fx) * 2 - 1

        pred_fy_pinn = ddpm_obtain_sr_img(n_fx, timesteps_test, p_model_pinn, inf_type='lang', smooth_factor=0.015,
                                          reg_type='l1')
        pred_fy_vanilla = ddpm_obtain_sr_img(n_fx, timesteps_test, p_model_vanilla, inf_type='lang', smooth_factor=0)
        ssim_pinn, ssim_vanilla, mae_pinn, mae_vanilla = plot_imgs(pred_fy_pinn, pred_fy_vanilla, fy, n_fx,
                                                                   experiment='shepp_logan')
        print('mean ssim pinn' + str(np.mean(ssim_pinn)))
        print('mean ssim vanilla' + str(np.mean(ssim_vanilla)))
        print('mean mae pinn' + str(np.mean(mae_pinn)))
        print('mean mae vanilla' + str(np.mean(mae_vanilla)))





    elif test == 'imagenet':
        data = np.load('./data/microscope_ds_imnet_test_noisy.npz')
        lam_vec = [0]
        n_imgs = 100
        for lam in lam_vec:
            print('lam: ' + str(lam))
            wf_images = data['widefield'][:n_imgs]
            KTest = np.expand_dims(data['widefield_kernel'][:n_imgs], -1)
            gt_images = np.expand_dims(data['object'][:n_imgs], -1)

            # wf_images = norm_01(wf_images) * 2 - 1
            #wf_images = norm_01(tf.abs(forward_model_conv([KTest, gt_images * 0.5 + 0.5]))) * 2 - 1 + np.random.normal(
            #    lam, lam, gt_images.shape)

            print(gt_images.shape)
            n_fx = norm_01(wf_images) * 2 - 1
            print(n_fx.shape)
            fy = norm_01(gt_images) * 2 - 1

            pred_fy_pinn = ddpm_obtain_sr_img(n_fx, timesteps_test, p_model_pinn, inf_type='lang',
                                              smooth_factor=0.015,
                                              reg_type='l1')
            pred_fy_vanilla = ddpm_obtain_sr_img(n_fx, timesteps_test, p_model_vanilla, inf_type='lang',
                                                 smooth_factor=0)
            ssim_pinn, ssim_vanilla, mae_pinn, mae_vanilla = plot_imgs(pred_fy_pinn, pred_fy_vanilla, fy, n_fx,
                                                                       experiment='imagenet_' + str(lam), slc=0)
            print('mean ssim pinn' + str(np.mean(ssim_pinn)))
            print('mean ssim vanilla' + str(np.mean(ssim_vanilla)))
            print('mean mae pinn' + str(np.mean(mae_pinn)))
            print('mean mae vanilla' + str(np.mean(mae_vanilla)))



    elif test == 'widefield':
        data = np.load('./c_w_patch_real.npz')
        XTest = norm_01(data['w_img'][..., 19:20]) * 2 - 1
        YTest = norm_01(data['c_img'][..., 19:20]) * 2 - 1

        objects = np.zeros_like(XTest)
        for idx in range(XTest.shape[0]):
            fx = XTest[idx:idx + 1]
            fy = YTest[idx:idx + 1]
            if np.amax(fy) >= 0:
                pred_fy = ddpm_obtain_sr_img(fx, timesteps_test, p_model_pinn, inf_type='lang', smooth_factor=0.01,
                                             reg_type='tv')
                plot_imgs(pred_fy, pred_fy, fy, fx, test, model_type=model_type, step=idx)
                objects[idx, :, :, 0] = pred_fy[0, :, :, 0]

        np.savez('./imgs_output/' + model_type + '_testing/reconstructions_widefield.npz', recon=objects)


    elif test == 'confocal':
        data = np.load('./c_w_patch_real.npz')

        XTest = data['c_img']
        YTest = data['c_img']
        objects = np.zeros_like(XTest)
        for idx in range(0, XTest.shape[0]):
            for slice_idx in range(14, 21):
                fx = norm_01(XTest[idx:idx + 1, :, :, slice_idx:slice_idx + 1]) * 2 - 1
                fy = norm_01(YTest[idx:idx + 1, :, :, slice_idx:slice_idx + 1]) * 2 - 1
                if np.amax(norm_01(YTest[..., 19:20])[idx:idx + 1] * 2 - 1) >= 0:
                    pred_fy = ddpm_obtain_sr_img(fx, timesteps_test, p_model_pinn, smooth_factor=0.015)
                    plot_imgs(pred_fy, pred_fy, fy, fx, test, model_type=model_type, step=idx, slc=slice_idx)
                    objects[idx, :, :, slice_idx] = pred_fy[0, :, :, 0] + fx[0, :, :, 0]

        np.savez('./imgs_output/' + model_type + '_testing/reconstructions_confocal.npz', recon=objects)


    elif test == 'biosr':
        data = np.load('./biosr_ds_test.npz')
        lam_vec = [0]
        n_imgs = 10
        offset = 9000
        for lam in lam_vec:
            print('lam: ' + str(lam))
            XTest = data['x'][offset:offset + n_imgs]
            YTest = data['y'][offset:offset + n_imgs]
            wf_images = XTest
            gt_images = YTest
            wf_images = tf.image.resize(wf_images, [patch_size, patch_size]).numpy()
            gt_images = tf.image.resize(gt_images, [patch_size, patch_size]).numpy()

            print(gt_images.shape)
            n_fx = norm_01(wf_images) * 2 - 1
            print(n_fx.shape)
            fy = norm_01(gt_images) * 2 - 1

            pred_fy_pinn = ddpm_obtain_sr_img(n_fx, timesteps_test, p_model_pinn, inf_type='lang',
                                              smooth_factor=0.015,
                                              reg_type='l1')
            pred_fy_vanilla = ddpm_obtain_sr_img(n_fx, timesteps_test, p_model_vanilla, inf_type='lang',
                                                 smooth_factor=0)
            ssim_pinn, ssim_vanilla, mae_pinn, mae_vanilla = plot_imgs(pred_fy_pinn, pred_fy_vanilla, fy, n_fx,
                                                                       experiment='biosr_' + str(lam), slc=0)
            print('mean ssim pinn' + str(np.mean(ssim_pinn)))
            print('mean ssim vanilla' + str(np.mean(ssim_vanilla)))
            print('mean mae pinn' + str(np.mean(mae_pinn)))
            print('mean mae vanilla' + str(np.mean(mae_vanilla)))

    else:
        n_samples = 1
        data = np.load('./data/biosr_ds_test.npz')
        XTest = data['x']
        YTest = data['y']
        ix = np.random.randint(0, YTest.shape[0], n_samples)
        offset = 9030
        fx = XTest[offset, :, :, 0]
        fx = tf.image.resize(np.expand_dims(np.expand_dims(fx, -1), 0), [patch_size, patch_size]).numpy()
        fy = tf.image.resize(YTest[offset:offset + n_samples], [patch_size, patch_size]).numpy()
        pred_fy = ddpm_obtain_sr_img(fx, timesteps_test, p_model_pinn, smooth_factor=0)
        plot_imgs(pred_fy, pred_fy, fy, fx, test, model_type=model_type)




test_sr_ddpm(900000, test='imagenet')
