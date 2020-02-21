import argparse

from model import *
from data import *
from data import get_data_paths_list
from data import _parse_data
import scipy.io as sio




parser = argparse.ArgumentParser()

parser.add_argument("--infer_data", default="data/infer")

FLAGS = parser.parse_args()


test_image_path = os.path.join(FLAGS.infer_data, 'linear')
test_mask_path = os.path.join(FLAGS.infer_data, 'full')


test_image_paths, test_mask_paths = get_data_paths_list(test_image_path, test_mask_path)
test_input, test_target = _parse_data(test_image_paths[0], test_mask_paths[0])


print(test_input.shape) #(1,100,256,256)


model = unet()

model.load_weights("unet_membrane.hdf5")
#results = model.evaluate(test_input,test_target, batch_size=1)
results = model.predict(test_input, batch_size = 1)

#sio.savemat(output_folder  + '/input_test.mat', {"input_test" : all_input_test})
#print('predictions shape:', results.shape)

sio.savemat('result/pred_test.mat', {"pred_test" : results})
