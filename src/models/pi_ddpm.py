import tensorflow as tf
from keras.layers import *
from tensorflow.keras.models import Model
from src.models.model import UNet_ddpm
from src.training_utils.forward_models import forward_model_conv
import tensorflow.keras.backend as K


def obtain_noisy_sample(x):
    x_0 = x[0]
    gamma = x[1]
    gamma_vec = tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1)
    noise_sample = tf.random.normal(tf.shape(x_0))
    return [tf.sqrt(gamma_vec) * x_0 + tf.sqrt(1 - gamma_vec) * noise_sample, noise_sample]


def obtain_clean_sample(x):
    pred_noise = x[0]
    gamma = x[1]
    x_t = x[2]
    gamma_vec = tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1)
    x_0 = (x_t - tf.sqrt(1 - gamma_vec) * pred_noise) / tf.clip_by_value(tf.sqrt(gamma_vec), 1e-5, 1)
    return x_0


def obtain_sr_t(x):
    x_t = x[0]
    x_lr = x[1]
    gamma = x[2]
    gamma_vec = tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1)
    y_t = x_t + tf.sqrt(gamma_vec) * x_lr
    return y_t


def norm_01(x):
    return tf.math.divide_no_nan(x - tf.stop_gradient(tf.math.reduce_min(x, axis=(1, 2, 3), keepdims=True)),
                                 tf.stop_gradient(tf.math.reduce_max(x, axis=(1, 2, 3), keepdims=True)) -
                                 tf.stop_gradient(tf.math.reduce_min(x, axis=(1, 2, 3), keepdims=True)))


def norm_2(x):
    return tf.reduce_sum(x ** 2, axis=(1, 2, 3), keepdims=True)


def forward_model_gradient(x):
    # lam = 1

    dirty_img = x[1]
    pred_dirty = norm_01(tf.image.resize(x[0], [K.int_shape(dirty_img)[1], K.int_shape(dirty_img)[2]])) * 2 - 1
    x_t = x[2]
    gamma = x[3]
    noise = x[4]
    gamma_vec = tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1)
    pi_l2 = tf.reduce_sum(tf.square(pred_dirty - dirty_img), axis=(1, 2, 3), keepdims=True)
    grad_pi = tf.gradients(pi_l2, x_t, unconnected_gradients='zero')[0]
    # lam = tf.math.reduce_max(grad_pi, axis=(1, 2, 3), keepdims=True)
    return grad_pi


def rescale(x):
    return tf.image.resize(x, [K.int_shape(x)[1] // 2, K.int_shape(x)[2] // 2]) * 0.5 + 0.5


def pi_model(input_shape_noisy, volume_shape, out_channels):
    noisy_input = Input(volume_shape)
    ref_frame = Input(input_shape_noisy)
    # ds_yt = Conv2D(32, 1, padding='same', strides=2)(noisy_input)
    # ds_ref = Conv2D(32, 1, padding='same', strides=2)(ref_frame)

    c_inpt = Concatenate()([noisy_input, ref_frame])
    gamma_inp = Input((1,))
    noise_out, _ = UNet_ddpm(input_shape_noisy, inputs=c_inpt, gamma_inp=gamma_inp,
                             out_filters=out_channels, z_enc=False)
    # m_dfcan = dfcan_ddpm(K.int_shape(c_inpt)[1:], out_filters=out_channels)
    # noise_out = m_dfcan([c_inpt, gamma_inp])

    model_out = Model([noisy_input, ref_frame, gamma_inp], noise_out)

    return model_out


def train_model(input_shape_condition, volume_shape, out_channels=1, noise_est=False):
    ground_truth = Input(volume_shape)
    dirty_img = Input(input_shape_condition)
    kernel_conv = Input((volume_shape[0] // 2, volume_shape[1] // 2, volume_shape[2]))
    gamma_inp = Input((1,))
    # ct_inp_shape = input_shape_noisy[:-1] + (input_shape_noisy[-1] + input_shape_noisy[-1],)
    # t_tiled = Lambda(tile_gamma)([t_inp, ground_truth])
    if noise_est:
        n_model = pi_model(input_shape_condition, volume_shape,
                           out_channels)
    else:
        n_model = pi_model(input_shape_condition, volume_shape,
                           out_channels + 1)  # adding the gradient as separate output to control nu outside of training

    n_sample = Lambda(obtain_noisy_sample)([ground_truth, gamma_inp])
    if not noise_est:
        noiseless_img = Input(input_shape_condition)
        noise_pred = n_model([n_sample[0], dirty_img, gamma_inp])
        delta_noise = n_sample[1] - noise_pred[..., 0:1]
        y_t = Lambda(obtain_sr_t)([n_sample[0], dirty_img, gamma_inp])
        pred_dirty = tf.abs(Lambda(forward_model_conv)([kernel_conv, Lambda(rescale)(y_t)]))
        grad = Lambda(forward_model_gradient)([pred_dirty, noiseless_img, y_t, gamma_inp, n_sample[1]])
        delta_grad = tf.stop_gradient(grad) - noise_pred[..., 1:]
        train_model = Model([ground_truth, dirty_img, noiseless_img, kernel_conv, gamma_inp], [delta_noise, delta_grad])
        train_model.summary()
    else:
        noise_pred = n_model([n_sample[0], dirty_img, gamma_inp])
        delta_noise = noise_pred - n_sample[1]
        train_model = Model([ground_truth, dirty_img, gamma_inp], delta_noise)
        train_model.summary()

    return n_model, train_model
