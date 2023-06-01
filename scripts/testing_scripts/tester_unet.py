from scripts.data_scripts.dataset_generation import norm_01
import numpy as np
from tensorflow_addons.layers import InstanceNormalization
import os
from keras.models import load_model
from natsort import natsorted
from src.training_utils.utils import dataGenerator_2Dstack


def test_unet(MODEL_PATH, MODEL_NAME, TEST_PATH, SAVE_PATH, BATCH='False'):
    # MODEL_PATH = './models_weight/'
    # SVAED_MODEL_NAME = 'model_Unet_imagenet_noise.hdf5'

    my_model = load_model(MODEL_PATH + MODEL_NAME, compile=False,
                          custom_objects={'InstanceNormalization': InstanceNormalization})

    # testing
    test_data_dir = TEST_PATH + '/'
    test_data_list = natsorted(os.listdir(test_data_dir))

    test_gen_class = dataGenerator_2Dstack(test_data_dir, test_data_list, 16)
    test_img_datagen = test_gen_class.imageLoader()

    if BATCH == 'True':
        # data generator
        _, img_test, o_test, _, _ = test_img_datagen.__next__()
        print('test in batches')

    elif BATCH == 'False':
        test_dataset = np.load(TEST_PATH + test_data_list[1])  # change to your testset
        img_test, o_test = test_dataset['w_img'], test_dataset['o']
        print('test all')

    else:
        print('wrong test set')

    img_test, o_test = norm_01(img_test), norm_01(o_test)
    pred_test = my_model.predict(img_test)

    np.savez(SAVE_PATH + 'test_results.npz', pred=pred_test, img=img_test, o=o_test)

    print('test dataset saved:', SAVE_PATH)


test_unet(MODEL_PATH='./models_weight/', MODEL_NAME='model_Unet.hdf5',
          TEST_PATH='./simu_biosr/test/', SAVE_PATH='./results', BATCH='False')
