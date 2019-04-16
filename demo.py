from helpers import center_crop
from fcn_model import fcn_model
import dicom
import cv2
import matplotlib.pyplot as plt
weights = 'rvsc_o_epoch_20.h5'

crop_size = (200,200,1)
num_class = 2

model = fcn_model(crop_size,num_class,weights)

# file name 'eval.dcm'
img_file = 'eval.dcm'

img = dicom.read_file(img_file)
img = img.pixel_array.astype('int32')
img = img.reshape(img.shape[0],img.shape[1],1)

img = center_crop(img,crop_size = 200)

plt.imshow(img.reshape((200,200)))
plt.show()


result = model.predict(img.reshape((1,200,200,1)))

plt.imshow(result.reshape((200,200)))
plt.show()



