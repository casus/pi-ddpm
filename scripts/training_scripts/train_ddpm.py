import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.models.pi_ddpm import train_model
import tensorflow.keras.backend as K
from skimage.util import montage
from src.training_utils.ddpm_utils import *
from src.training_utils.forward_models import forward_model_conv
from tester_diffusion import norm_2


def norm_01(x):
    return np.nan_to_num((x - np.amin(x, axis=(1, 2, 3), keepdims=True)) / (
            np.amax(x, axis=(1, 2, 3), keepdims=True) - np.amin(x, axis=(1, 2, 3), keepdims=True)))


def train_pinn_ddpm(type='widefield'):
    lr = 1e-4
    opt_m = tf.keras.optimizers.Adam(lr)
    patch_size = 256
    batch_size = 4
    timesteps = 2000
    timesteps_test = 200
    train_steps = int(2e5)
    ds_sample = int(1e4)
    print_freq = int(1e3)
    lr_change_ts = int(4e5)
    img_channels = 1

    gamma_vec_t, alpha_vec = variance_schedule(timesteps, schedule_type='cos')
    p_model, t_model = train_model((patch_size, patch_size, 1), (patch_size, patch_size, img_channels), img_channels,
                                   noise_est=False)

    t_model.compile(loss='mse', loss_weights=[1, 0.01], optimizer=opt_m)
    #Use this to load your data, it can be BioSR, ImageNet, etc. Our strategy is to pretrain on imagenet, then finetune for a few iterations (always using synthetic data)
    #on your target dataset, one example Imagenet -> BioSR.
    data_biosr = np.load('/media/gabriel/data_hdd/biosr_dataset/train/biosr_ds.npz')
    print('loading biosr')
    YBiosr = data_biosr['y']

    for step in range(train_steps):
        if step % ds_sample == 0:
            b_idx = np.random.randint(0, 3)
            data = np.load('/media/gabriel/data_hdd/microscope_ds_imnet_' + str(b_idx) + '.npz')
            if type == 'widefield':
                KTrain = np.expand_dims(data['widefield_kernel'], -1)
            else:
                KTrain = np.expand_dims(data['confocal_kernel'], -1)

            print('current learning rate: ' + str(t_model.optimizer.learning_rate.numpy()))
            avg_loss = 0
        ix = np.random.randint(0, YBiosr.shape[0], batch_size)
        kx = np.random.randint(0, KTrain.shape[0], batch_size)
        fy_b = tf.image.resize(YBiosr[ix], [patch_size, patch_size]).numpy()
        fy = norm_01(fy_b) * 2 - 1
        fy_ds = norm_01(tf.image.resize(fy_b, [patch_size // 2, patch_size // 2]).numpy()) * 2 - 1
        fk = KTrain[kx]

        fx = norm_01(tf.abs(forward_model_conv([fk, fy_ds * 0.5 + 0.5]))) * 2 - 1
        fx = tf.image.resize(fx, [patch_size, patch_size]).numpy()
        lam = np.random.uniform(0.001, 0.1)
        n_fx = norm_01(fx) + np.random.normal(lam, lam, fx.shape)
        n_fx = norm_01(n_fx) * 2 - 1
        dfy = fy - n_fx

        gamma_vec = np.zeros((batch_size, 1))

        for idx in range(batch_size):
            gamma_vec[idx] = sample_gamma(gamma_vec_t)

        loss = t_model.train_on_batch([dfy, n_fx, fx, fk, gamma_vec], [np.zeros_like(fy), np.zeros_like(fy)])

        loss_flow = np.array(loss)
        avg_loss += 1 / (step % ds_sample + 1) * (loss_flow - avg_loss)

        if step == lr_change_ts:
            print('changing learning rate to: ' + str((t_model.optimizer.learning_rate * 0.5).numpy()))
            K.set_value(t_model.optimizer.learning_rate, t_model.optimizer.learning_rate * 0.5)

        print('step: ' + str(step) + 'loss composite: ' + str(avg_loss))
        if step % print_freq == 0:
            pred_sr = np.random.normal(0, 1, fy.shape)
            gamma_vec_test, alpha_vec_test = variance_schedule(timesteps_test, schedule_type='linear')
            for t in tqdm(range(timesteps_test, 0, -1)):
                z = np.random.normal(0, 1, fy.shape)
                if t == 1:
                    z = 0
                alpha_t = alpha_vec_test[t - 1]
                gamma_t = gamma_vec_test[t - 1]
                gamma_tm1 = gamma_vec_test[t - 2]
                beta_t = 1 - alpha_t

                gamma_t_inp = np.ones_like(gamma_vec) * np.reshape(gamma_t, (1, 1))
                pred_params = p_model.predict([pred_sr, n_fx, gamma_t_inp], verbose=False)
                if pred_params.shape[-1] == 1:
                    pred_noise = pred_params
                    pred_gradient = 0
                else:

                    pred_noise = pred_params[..., 0:1]
                    pred_gradient = pred_params[..., 1:]
                    pred_gradient = pred_gradient * 10 * np.sqrt(gamma_t) / norm_2(pred_gradient)

                alpha_factor = beta_t / np.sqrt(1 - gamma_t)
                beta_factor = (1 - gamma_tm1) * beta_t / (1 - gamma_t)
                pred_sr = 1 / np.sqrt(alpha_t) * (pred_sr - alpha_factor * pred_noise) + np.sqrt(
                    beta_factor) * z
                if t == timesteps_test // 2:
                    grad_h = pred_gradient

            int_res = norm_01(pred_sr + n_fx) * 2 - 1
            int_dfy = norm_01(pred_sr) * 2 - 1
            int_gt = norm_01(fy) * 2 - 1
            int_x = norm_01(n_fx) * 2 - 1
            int_grad_h = norm_01(grad_h) * 2 - 1

            c_imgs_result = montage(
                np.squeeze(np.concatenate((int_res, int_dfy, int_gt, int_x, int_grad_h), axis=2)))

            plt.imsave('/home/gabriel/Documents/dl_projects/pi-ddpm/imgs_output/img_sr' + str(step) + '.png',
                       np.uint8(np.clip(c_imgs_result, -1, 1) * 127.5 + 127.5))

        if step % 20000 == 0:
            t_model.save_weights(
                '/home/gabriel/Documents/dl_projects/pi-ddpm/models_weights/df_ddpm/model' + str(step) + '.h5')
            # log_weights(n_logger, './models_dfy/model' + str(step) + '.h5')


# change type from widefield to confocal to use widefield microscope psfs or confocal psfs
train_pinn_ddpm(type='widefield')
