import matplotlib.pyplot as plt
import tensorflow as tf
from src.models.pi_ddpm import train_model
import tensorflow.keras.backend as K

from skimage.util import montage
from src.training_utils.ddpm_utils import *
from src.training_utils.forward_models import forward_model_conv


def norm_01(x):
    return np.nan_to_num((x - np.amin(x, axis=(1, 2, 3), keepdims=True)) / (
            np.amax(x, axis=(1, 2, 3), keepdims=True) - np.amin(x, axis=(1, 2, 3), keepdims=True)))


def train_pinn_ddpm(type='widefield'):
    lr = 1e-4
    opt_m = tf.keras.optimizers.Adam(lr)
    patch_size = 128
    # K.set_floatx('float32')
    batch_size = 8
    timesteps = 2000
    timesteps_test = 200
    train_steps = int(1.2e6)
    ds_sample = int(1e4)
    print_freq = int(1e3)
    lr_change_ts = int(4e5)
    img_channels = 1

    gamma_vec_t, alpha_vec = variance_schedule(timesteps, schedule_type='cos')
    p_model, t_model = train_model((patch_size, patch_size, 1), (patch_size, patch_size, img_channels), img_channels,
                                   noise_est=False)
    t_model.compile(loss='mse', loss_weights=[1, 0], optimizer=opt_m)
    # t_model.compile(loss='mse', optimizer=opt_m)
    YTrain = np.load('./data/imnet_imgs_full.npz')['imgs']
    data_biosr = np.load('./data/biosr_ds.npz')
    YBiosr = data_biosr['y']

    for step in range(train_steps):
        if step % ds_sample == 0:
            b_idx = np.random.randint(0, 3)
            data = np.load('./data/microscope_ds_imnet_' + str(b_idx) + '.npz')
            if type == 'widefield':
                KTrain = np.expand_dims(data['widefield_kernel'], -1)
            else:
                KTrain = np.expand_dims(data['confocal_kernel'], -1)
            # KTrain = np.nan_to_num(KTrain / np.amax(KTrain, axis=(1, 2, 3), keepdims=True), nan=1e-4)

            # XTrain = np.array(tf.abs(forward_model_conv([KTrain, YTrain * 0.5 + 0.5])))
            # print(np.amax(XTrain))

            print('current learning rate: ' + str(t_model.optimizer.learning_rate.numpy()))
            avg_loss = 0
        ix = np.random.randint(0, YBiosr.shape[0], batch_size // 2)
        tx = np.random.randint(0, YTrain.shape[0], batch_size // 2)
        kx = np.random.randint(0, KTrain.shape[0], batch_size)

        fy_i = tf.image.resize(YTrain[tx], [patch_size, patch_size]).numpy()

        fy_b = tf.image.resize(YBiosr[ix], [patch_size, patch_size]).numpy()
        fy = np.concatenate((fy_i, fy_b), axis=0)
        # fy = fy_b
        # fy = random_erase(fy, 0.5)
        fk = KTrain[kx]

        fx = norm_01(tf.abs(forward_model_conv([fk, fy * 0.5 + 0.5]))) * 2 - 1
        lam = np.random.uniform(0.001, 0.05)
        n_fx = norm_01(fx) + np.random.normal(lam, lam, fx.shape)
        n_fx = norm_01(n_fx) * 2 - 1
        dfy = fy - n_fx

        gamma_vec = np.zeros((batch_size, 1))

        for idx in range(batch_size):
            gamma_vec[idx] = sample_gamma(gamma_vec_t)
        # gamma_exp = np.expand_dims(np.expand_dims(gamma_vec, -1), -1) * np.ones_like(fy)

        loss = t_model.train_on_batch([dfy, n_fx, fx, fk, gamma_vec], [np.zeros_like(fy), np.zeros_like(fy)])
        # loss = t_model.train_on_batch([dfy, n_fx, gamma_vec], np.zeros_like(fy))

        loss_flow = np.array(loss)
        avg_loss += 1 / (step % ds_sample + 1) * (loss_flow - avg_loss)

        if step == lr_change_ts:
            print('changing learning rate to: ' + str((t_model.optimizer.learning_rate * 0.5).numpy()))
            K.set_value(t_model.optimizer.learning_rate, t_model.optimizer.learning_rate * 0.5)

        print('step: ' + str(step) + 'loss composite: ' + str(avg_loss))
        if step % print_freq == 0:
            pred_sr = np.random.normal(0, 1, fy.shape)
            gamma_vec_test, alpha_vec_test = variance_schedule(timesteps_test, schedule_type='linear')
            # t_ft = np.expand_dims(np.expand_dims(ft, axis=-1), axis=-1) * np.ones_like(pred_sr)
            for t in tqdm(range(timesteps_test, 0, -1)):
                z = np.random.normal(0, 1, fy.shape)
                if t == 1:
                    z = 0
                alpha_t = alpha_vec_test[t - 1]
                gamma_t = gamma_vec_test[t - 1]
                gamma_tm1 = gamma_vec_test[t - 2]
                beta_t = 1 - alpha_t

                gamma_t_inp = np.ones_like(gamma_vec) * np.reshape(gamma_t, (1, 1))
                pred_noise = p_model.predict([pred_sr, n_fx, gamma_t_inp], verbose=False)
                # pred_noise = (pred_sr - np.sqrt(gamma_t) * x_0) / np.sqrt(1 - gamma_t)

                alpha_factor = beta_t / np.sqrt(1 - gamma_t)
                beta_factor = (1 - gamma_tm1) * beta_t / (1 - gamma_t)
                pred_sr = 1 / np.sqrt(alpha_t) * (pred_sr - alpha_factor * pred_noise) + np.sqrt(beta_t) * z

            int_res = np.clip(pred_sr[..., 0] + n_fx[..., 0], -1, 1)
            int_dfy = np.clip(pred_sr[..., 0], -1, 1)
            int_gt = np.clip(fy[..., 0], -1, 1)
            int_x = np.clip(n_fx[..., 0], -1, 1)

            c_imgs_result = montage(np.squeeze(np.concatenate((int_res, int_dfy, int_gt, int_x), axis=2)))

            plt.imsave('./imgs_output/img_sr' + str(step) + '.png',
                       np.uint8(np.clip(c_imgs_result, -1, 1) * 127.5 + 127.5))

        if step % 2000 == 0:
            t_model.save_weights('./models_weights/pinn_finetuned/model' + str(step) + '.h5', True)
            # log_weights(n_logger, './models_dfy/model' + str(step) + '.h5')

#change type from widefield to confocal to use widefield microscope psfs or confocal psfs
train_pinn_ddpm(type='widefield')
