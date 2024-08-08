import matplotlib.pyplot as plt
import tensorflow as tf
from src.models.model import UNet
import tensorflow.keras.backend as K

from skimage.util import montage
from src.training_utils.ddpm_utils import *
from src.training_utils.forward_models import forward_model_conv


def norm_01(x):
    return np.nan_to_num((x - np.amin(x, axis=(1, 2, 3), keepdims=True)) / (
            np.amax(x, axis=(1, 2, 3), keepdims=True) - np.amin(x, axis=(1, 2, 3), keepdims=True)))


def train_unet(type='widefield'):
    lr = 1e-4
    opt_m = tf.keras.optimizers.Adam(lr)
    patch_size = 128
    # K.set_floatx('float32')
    batch_size = 8
    train_steps = int(1.2e6)
    ds_sample = int(1e4)
    print_freq = int(1e3)
    lr_change_ts = int(4e5)
    img_channels = 1

    t_model = UNet((patch_size, patch_size, img_channels), out_channels=img_channels)
    t_model.compile(loss='mae', optimizer=opt_m)

    data_biosr = np.load('/media/gabriel/data_hdd/biosr_dataset/biosr_ds.npz')
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
        fy = fy_b
        fy = norm_01(fy) * 2 - 1
        fk = KTrain[kx]


        fx = norm_01(tf.abs(forward_model_conv([fk, fy * 0.5 + 0.5]))) * 2 - 1
        lam = np.random.uniform(0.001, 0.05)
        n_fx = norm_01(fx) + np.random.normal(lam, lam, fx.shape)
        n_fx = norm_01(n_fx) * 2 - 1
        dfy = fy-n_fx
        loss = t_model.train_on_batch(n_fx, dfy)
        # loss = t_model.train_on_batch([dfy, n_fx, gamma_vec], np.zeros_like(fy))

        loss_flow = np.array(loss)
        avg_loss += 1 / (step % ds_sample + 1) * (loss_flow - avg_loss)

        if step == lr_change_ts:
            print('changing learning rate to: ' + str((t_model.optimizer.learning_rate * 0.5).numpy()))
            K.set_value(t_model.optimizer.learning_rate, t_model.optimizer.learning_rate * 0.5)

        print('step: ' + str(step) + 'loss composite: ' + str(avg_loss))
        if step % print_freq == 0:
            pred_sr = t_model.predict(n_fx)
            int_res = np.clip(pred_sr[..., 0] + n_fx[..., 0], -1, 1)
            int_dfy = np.clip(pred_sr[..., 0], -1, 1)
            int_gt = np.clip(fy[..., 0], -1, 1)
            int_x = np.clip(n_fx[..., 0], -1, 1)

            c_imgs_result = montage(np.squeeze(np.concatenate((int_res, int_dfy, int_gt, int_x), axis=2)))

            plt.imsave('./imgs_output/img_sr' + str(step) + '.png',
                       np.uint8(np.clip(c_imgs_result, -1, 1) * 127.5 + 127.5))

        if step % 5000 == 0:
            t_model.save_weights('./models_weights/unet/model' + str(step) + '.h5', True)
            # log_weights(n_logger, './models_dfy/model' + str(step) + '.h5')


train_unet(type='widefield')
