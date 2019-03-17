import os
import numpy as np
import re
import dicom
import cv2
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import Model
from keras import optimizers
import keras.backend as K

# Create Dataset
class Contour(object): 
    # contour object to save information's contour file
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'P(\d{02})-(\d{04})-.*', ctr_path)
        self.patient_no = match.group(1)
        self.img_no = match.group(2)

class CreatData():
    def __init__(self, data_path,contour_type):
        self.path = data_path
        self.contour_type = contour_type

    def center_crop(self,data,crop_size):
        h,w,d = data.shape
        
        if any([dim < crop_size for dim in (h, w)]):
            pad_h = (crop_size - h) if h < crop_size else 0
            pad_w = (crop_size - w) if w < crop_size else 0

            rem_h = pad_h % 2
            rem_w = pad_w % 2

            data = np.pad(data,((pad_h/2,pad_h/2+rem_h),(pad_w/2,pad_w/2+rem_w),(0,0)),
                                'constant',constant_values=0)
            h,w,d = data.shape
        
        h_offset = int((h-crop_size)/2)
        w_offset = int((w-crop_size)/2)

        newData = data[h_offset:h_offset+crop_size,w_offset:w_offset+crop_size,:]
        return newData

    def read_contour(self,contour):
        # read dicom image and coords to create mask
        img_path = [d 
            for d,n,f in os.walk(self.path)
                if contour.patient_no+'dicom' in d][0]
        
        file_name = 'P{}-{}.dcm'.format(contour.patient_no,contour.img_no)
        fullpath = os.path.join(img_path,file_name)

        f = dicom.read_file(fullpath)
        img = f.pixel_array.astype('int')
        if img.ndim < 3:
            img = img[...,np.newaxis]
        
        mask = np.zeros_like(img,dtype='int')
        coords = np.loadtxt(contour.ctr_path, delimiter= ' ').astype('int')
        # lấp đầy khu vực của RV được giới hạn bởi các coords
        cv2.fillPoly(mask,[coords],1)

        if mask.ndim < 3:
            mask = mask[..., np.newaxis]

        return img,mask


    def export_data(self, crop_size, shuffle=True):
        # đường dẫn tới file ex: P0list.txt
        list_file = [ os.path.join(d,e) 
            for d,n,f in os.walk(self.path) 
                for e in f if 'list' in e ]

        # đường dẫn tới file chứa tọa độ gắn nhãn
        contour = []

        for f in list_file:
            for line in open(f).readlines():
                line =  line.strip().replace('\\','/') # \ -> /
                full_path = os.path.join(self.path,line)
                if self.contour_type+'contour' in full_path:
                    contour.append(full_path)
        
        if shuffle:
            np.random.shuffle(contour)
        
        # tạo đối tượng Contour lưu thông tin về contour
        contourList = []
        for i in contour:
            cont = Contour(i)
            contourList.append(cont)

        # đọc ảnh và contour -> tạo dataset
        images = np.zeros((len(contourList),crop_size,crop_size,1))
        masks = np.zeros((len(contourList),crop_size,crop_size,1))
        
        for index,cont in enumerate(contourList):
            img,mask = self.read_contour(cont)
            img = self.center_crop(img,crop_size)
            mask = self.center_crop(mask,crop_size)
            
            images[index] = img
            masks[index] = mask
        
        return images, masks
        

# Create Model

class FCN_Model():
    def __init__(self,input_shape,num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def mvn(self,tensor):
        # mean-variance normalization
        '''
        dim's tensor is: None, h, w ,d
        '''
        ep = 1e-6
        mean = K.mean(tensor,axis=(1,2),keepdims=True)
        std = K.std(tensor,axis=(1,2),keepdims=True)

        mvn = (tensor-mean)/(std+ep)
        return mvn
    def crop(self,tensor):
        # danh sách 2 tensor, tensor thứ 2 có chiều lớn, đưa chúng cùng về chiều

        h_dims, w_dims = [],[]
        for t in tensor:
            n,h,w,d = K.get_variable_shape(t)
            h_dims.append(h)
            w_dims.append(w)
        pad_h = h_dims[1] - h_dims[0]
        pad_W = w_dims[1] - w_dims[0]

        crop_h_dim = (pad_h/2,pad_h/2+pad_h%2)
        crop_w_dim = (pad_W/2,pad_W/2+pad_W%2)

        croppped = Cropping2D(cropping=(crop_h_dim,crop_w_dim))(tensor[1])
        return croppped

    def dice_coef(self,y_true,y_pre,smooth=0.0):
        axis = (1,2)
        inter = K.sum(y_pre*y_true,axis=axis,keepdims=True)
        summation = K.sum(y_pre,axis=axis,keepdims=True) + K.sum(y_true,axis=axis,keepdims=True)

        return K.mean((2*inter + smooth)/(summation + smooth),axis=0)

    def dice_coef_loss(self,y_true,y_pre):
        return 1.0 - dice_coef(y_true,y_pre)
    def create_model(self,weight=None):
        if self.num_classes == 2:
            self.num_classes = 1
            loss = self.dice_coef_loss
            activation = 'sigmoid'
        else:
            loss = 'categorical_crossentropy'
            activation = 'softmax'

        kwargs = dict(
            kernel_size=3,
            strides=1,
            activation='relu',
            padding='same',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
        )
        
        data = Input(shape=self.input_shape, dtype='float', name='data')
        mvn0 = Lambda(self.mvn, name='mvn0')(data)
        pad = ZeroPadding2D(padding=5, name='pad')(mvn0)

        conv1 = Conv2D(filters=64, name='conv1', **kwargs)(pad)
        mvn1 = Lambda(self.mvn, name='mvn1')(conv1)
        
        conv2 = Conv2D(filters=64, name='conv2', **kwargs)(mvn1)
        mvn2 = Lambda(self.mvn, name='mvn2')(conv2)

        conv3 = Conv2D(filters=64, name='conv3', **kwargs)(mvn2)
        mvn3 = Lambda(self.mvn, name='mvn3')(conv3)
        pool1 = MaxPooling2D(pool_size=3, strides=2,
                        padding='valid', name='pool1')(mvn3)

        
        conv4 = Conv2D(filters=128, name='conv4', **kwargs)(pool1)
        mvn4 = Lambda(self.mvn, name='mvn4')(conv4)

        conv5 = Conv2D(filters=128, name='conv5', **kwargs)(mvn4)
        mvn5 = Lambda(self.mvn, name='mvn5')(conv5)

        conv6 = Conv2D(filters=128, name='conv6', **kwargs)(mvn5)
        mvn6 = Lambda(self.mvn, name='mvn6')(conv6)

        conv7 = Conv2D(filters=128, name='conv7', **kwargs)(mvn6)
        mvn7 = Lambda(self.mvn, name='mvn7')(conv7)
        pool2 = MaxPooling2D(pool_size=3, strides=2,
                        padding='valid', name='pool2')(mvn7)


        conv8 = Conv2D(filters=256, name='conv8', **kwargs)(pool2)
        mvn8 = Lambda(self.mvn, name='mvn8')(conv8)

        conv9 = Conv2D(filters=256, name='conv9', **kwargs)(mvn8)
        mvn9 = Lambda(self.mvn, name='mvn9')(conv9)

        conv10 = Conv2D(filters=256, name='conv10', **kwargs)(mvn9)
        mvn10 = Lambda(self.mvn, name='mvn10')(conv10)

        conv11 = Conv2D(filters=256, name='conv11', **kwargs)(mvn10)
        mvn11 = Lambda(self.mvn, name='mvn11')(conv11)
        pool3 = MaxPooling2D(pool_size=3, strides=2,
                        padding='valid', name='pool3')(mvn11)
        drop1 = Dropout(rate=0.5, name='drop1')(pool3)


        conv12 = Conv2D(filters=512, name='conv12', **kwargs)(drop1)
        mvn12 = Lambda(self.mvn, name='mvn12')(conv12)

        conv13 = Conv2D(filters=512, name='conv13', **kwargs)(mvn12)
        mvn13 = Lambda(self.mvn, name='mvn13')(conv13)

        conv14 = Conv2D(filters=512, name='conv14', **kwargs)(mvn13)
        mvn14 = Lambda(self.mvn, name='mvn14')(conv14)

        conv15 = Conv2D(filters=512, name='conv15', **kwargs)(mvn14)
        mvn15 = Lambda(self.mvn, name='mvn15')(conv15)
        drop2 = Dropout(rate=0.5, name='drop2')(mvn15)


        score_conv15 = Conv2D(filters=self.num_classes, kernel_size=1,
                            strides=1, activation=None, padding='valid',
                            kernel_initializer='glorot_uniform', use_bias=True,
                            name='score_conv15')(drop2)
        upsample1 = Conv2DTranspose(filters=self.num_classes, kernel_size=3,
                            strides=2, activation=None, padding='valid',
                            kernel_initializer='glorot_uniform', use_bias=False,
                            name='upsample1')(score_conv15)
        score_conv11 = Conv2D(filters=self.num_classes, kernel_size=1,
                            strides=1, activation=None, padding='valid',
                            kernel_initializer='glorot_uniform', use_bias=True,
                            name='score_conv11')(mvn11)
        crop1 = Lambda(self.crop, name='crop1')([upsample1, score_conv11])
        fuse_scores1 = average([crop1, upsample1], name='fuse_scores1')
        
        upsample2 = Conv2DTranspose(filters=self.num_classes, kernel_size=3,
                            strides=2, activation=None, padding='valid',
                            kernel_initializer='glorot_uniform', use_bias=False,
                            name='upsample2')(fuse_scores1)
        score_conv7 = Conv2D(filters=self.num_classes, kernel_size=1,
                            strides=1, activation=None, padding='valid',
                            kernel_initializer='glorot_uniform', use_bias=True,
                            name='score_conv7')(mvn7)
        crop2 = Lambda(self.crop, name='crop2')([upsample2, score_conv7])
        fuse_scores2 = average([crop2, upsample2], name='fuse_scores2')
        
        upsample3 = Conv2DTranspose(filters=self.num_classes, kernel_size=3,
                            strides=2, activation=None, padding='valid',
                            kernel_initializer='glorot_uniform', use_bias=False,
                            name='upsample3')(fuse_scores2)
        crop3 = Lambda(self.crop, name='crop3')([data, upsample3])
        predictions = Conv2D(filters=self.num_classes, kernel_size=1,
                            strides=1, activation=activation, padding='valid',
                            kernel_initializer='glorot_uniform', use_bias=True,
                            name='predictions')(crop3)
        
        model = Model(inputs=data, outputs=predictions)
        if weights is not None:
            model.load_weights(weights)
        sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss=loss,
                    metrics=['accuracy', self.dice_coef, self.jaccard_coef])

        return model



def main():
    data_path = os.path.join('RVSC_data','TrainingSet')

    data = CreatData(data_path,'i')

    
    crop_size = 200

    img_train, mask_train = data.export_data(crop_size)

    
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    epochs = 20
    mini_batch_size = 1

    image_generator = image_datagen.flow(img_train, shuffle=False,
                                    batch_size=mini_batch_size)
    mask_generator = mask_datagen.flow(mask_train, shuffle=False,
                                    batch_size=mini_batch_size)
    train_generator = izip(image_generator, mask_generator)

    input_shape = (100,100,1)
    num_classes = 2
    model = FCN_Model(input_shape,num_classes)

    model = model.create_model()

    for e in range(epochs):
        print('\nMain Epoch {:d}\n'.format(e+1))
        print('\nLearning rate: {:6f}\n'.format(lrate))
        train_result = []
        for iteration in range(len(img_train)/mini_batch_size):
            img, mask = next(train_generator)
            res = model.train_on_batch(img, mask)
            curr_iter += 1
            lrate = lr_poly_decay(model, base_lr, curr_iter,
                                  max_iter, power=0.5)
            train_result.append(res)
        train_result = np.asarray(train_result)
        train_result = np.mean(train_result, axis=0).round(decimals=10)
        print('Train result {:s}:\n{:s}'.format(model.metrics_names, train_result))
    pass
main()
        
    
