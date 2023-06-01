import tensorflow as tf
from keras.layers import *
from tensorflow.keras.models import Model
from model import UNet_ddpm
from src.training_utils.forward_models import forward_model_conv


def obtain_noisy_sample(x):
    x_0 = x[0]
    gamma = x[1]
    gamma_vec = tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1)
    noise_sample = tf.random.normal(tf.shape(x_0))
    return [tf.sqrt(gamma_vec) * x_0 + tf.sqrt(1 - gamma_vec) * noise_sample, noise_sample]


def obtain_sr_t(x):
    x_t = x[0]
    x_lr = x[1]
    gamma = x[2]
    gamma_vec = tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1)
    y_t = x_t + tf.sqrt(gamma_vec) * x_lr
    return y_t


def norm_01(x):
    return tf.math.divide_no_nan(x - tf.math.reduce_min(x, axis=(1, 2, 3), keepdims=True),
                                 tf.math.reduce_max(x, axis=(1, 2, 3), keepdims=True) -
                                 tf.math.reduce_min(x, axis=(1, 2, 3), keepdims=True))


def forward_model_gradient(x):
    lam = 1
    pred_dirty = norm_01(tf.abs(x[0])) * 2 - 1
    dirty_img = x[1]
    x_t = x[2]
    gamma = x[3]
    gamma_vec = tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1)
    pi_l2 = tf.math.reduce_mean(tf.square(pred_dirty - dirty_img), axis=(1, 2, 3), keepdims=True)
    grad_pi = tf.gradients(pi_l2, x_t, unconnected_gradients='zero')[0]
    # you can have lambda * grad_pi to increase the strength of the gradient.
    return lam * grad_pi


def rescale(x):
    return x * 0.5 + 0.5


def pi_model(input_shape_noisy, volume_shape, out_channels):
    noisy_input = Input(volume_shape)
    ref_frame = Input(input_shape_noisy)

    c_inpt = Concatenate()([noisy_input, ref_frame])
    gamma_inp = Input((1,))
    noise_out, _ = UNet_ddpm(input_shape_noisy, inputs=c_inpt, gamma_inp=gamma_inp,
                             out_filters=out_channels, z_enc=False)

    model_out = Model([noisy_input, ref_frame, gamma_inp], noise_out)

    return model_out


def train_model(input_shape_condition, volume_shape, out_channels=1, noise_est=False):
    ground_truth = Input(volume_shape)
    dirty_img = Input(input_shape_condition)
    kernel_conv = Input(volume_shape)
    gamma_inp = Input((1,))
    # ct_inp_shape = input_shape_noisy[:-1] + (input_shape_noisy[-1] + input_shape_noisy[-1],)
    # t_tiled = Lambda(tile_gamma)([t_inp, ground_truth])
    n_model = pi_model(input_shape_condition, volume_shape, out_channels)
    n_sample = Lambda(obtain_noisy_sample)([ground_truth, gamma_inp])
    if not noise_est:
        noiseless_img = Input(input_shape_condition)
        noise_pred = n_model([n_sample[0], dirty_img, gamma_inp])
        delta_noise = n_sample[1] - noise_pred
        y_t = Lambda(obtain_sr_t)([n_sample[0], dirty_img, gamma_inp])
        pred_dirty = Lambda(forward_model_conv)([kernel_conv, Lambda(rescale)(y_t)])
        grad = Lambda(forward_model_gradient)([pred_dirty, noiseless_img, n_sample[0], gamma_inp])
        delta_noise += grad

        train_model = Model([ground_truth, dirty_img, noiseless_img, kernel_conv, gamma_inp], [delta_noise, grad])
        train_model.summary()
    else:
        noise_pred = n_model([n_sample[0], dirty_img, gamma_inp])
        delta_noise = noise_pred - n_sample[1]
        train_model = Model([ground_truth, dirty_img, gamma_inp], delta_noise)
        train_model.summary()

    return n_model, train_model
