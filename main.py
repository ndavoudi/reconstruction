from model import *
from data import get_data_paths_list
from data import _parse_data
import argparse
import os
import time
from keras.preprocessing import sequence
from matplotlib import pyplot as plt

from keras.callbacks import EarlyStopping



parser = argparse.ArgumentParser()

#parser.add_argument("--mode", default="infer")
parser.add_argument("--train_data", default="data/training", help="Directory for training images")
parser.add_argument("--val_data", default="data/validation", help="Directory for validation images")
#parser.add_argument("--ckpt", default="models/model.ckpt", help="Directory for storing model checkpoints")
#parser.add_argument("--batch_size", default=2, help="Batch size for use in training", type=int)
#parser.add_argument("--infer_data", default="data/infer")
#parser.add_argument("--output_folder", default="data/output")
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

maxlen = 100

def main():





    start_time =  time.time()

    FLAGS = parser.parse_args()

    train_image_path = os.path.join(FLAGS.train_data, 'linear')
    train_mask_path = os.path.join(FLAGS.train_data, 'full')

    val_image_path = os.path.join(FLAGS.val_data, 'linear')
    val_mask_path = os.path.join(FLAGS.val_data, 'full')



    train_image_paths, train_mask_paths = get_data_paths_list(train_image_path, train_mask_path)
    train_input, train_target = _parse_data(train_image_paths[0], train_mask_paths[0])


    val_image_paths, val_mask_paths = get_data_paths_list(val_image_path, val_mask_path)
    val_input, val_target = _parse_data(val_image_paths[0], val_mask_paths[0])




    #train_input = sequence.pad_sequences(train_input, maxlen=maxlen)
    #train_target = sequence.pad_sequences(train_target, maxlen=maxlen)



    print(train_input.shape) #(1,100,256,256)
    print(train_target.shape) #(1,100,256,256)



    #val_image_paths, val_mask_paths = data.get_data_paths_list(val_image_path, val_mask_path)
    #val_input, val_target = data._parse_data(val_image_paths[0], val_mask_paths[0])

    #test_image_paths, test_mask_paths = data.get_data_paths_list(test_image_path, test_mask_path)
    #test_input, test_target = data._parse_data(test_image_paths[0], test_mask_paths[0])


    model = unet()

    for layer in model.layers:
        print(layer.output_shape)

    model.load_weights('unet_membrane.hdf5') # load trained model 


    #callbacks = [K.EarlyStopping(monitor='val_loss', min_delta=1e-2,patience=2,verbose=1)]
    #es = EarlyStopping(monitor='val_mse', mode='min', verbose=1, patience=10)



    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',save_best_only=True) # early stopping
    history = model.fit(train_input, train_target, batch_size = 4, epochs=2,verbose=2, callbacks=[model_checkpoint],  validation_data=(val_input, val_target))
    #history = model.fit(train_input, train_target, batch_size = 5, epochs=15,verbose=2, callbacks=[es],  validation_data=(val_input, val_target))


    #model.save_weights('./checkpoints/my_checkpoint')


    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])

    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('result/mse_train')
    plt.close()

    plt.plot(history.history['ssim_loss'])
    plt.plot(history.history['val_ssim_loss'])
    plt.xlabel('epoch')
    plt.ylabel('ssim')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('result/ssim_train')
    #plt.show()
    plt.close()

    # The returned "history" object holds a record
    # of the loss values and metric values during training
    #print('\nhistory dict:', history.history)

    '''
    del history
    K.clear_session()
    gc.collect()
    '''






if __name__ == "__main__":
    main()
