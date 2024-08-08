import tensorflow as tf


def forward_model_conv(x):
    conv_kernel = tf.cast(x[0][..., 0], dtype=tf.complex64)
    pred_img = tf.cast(x[1][..., 0], dtype=tf.complex64)


    ft_conv_kernel = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(conv_kernel)))
    ft_pred_img = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(pred_img)))

    conv_img_ft = ft_conv_kernel * ft_pred_img

    out_img_d = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(conv_img_ft)))

    return tf.expand_dims(out_img_d, -1)
