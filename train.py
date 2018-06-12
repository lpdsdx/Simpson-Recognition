import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle#将数据转化为文件保存在磁盘并可以再次读取
import h5py
import glob#查找符合特定规则的文件路径名
import time
from random import shuffle#将序列的所有元素随机排序
from collections import Counter#便捷快速计数
import matplotlib.pyplot as plt
import math
import itertools#笛卡尔积

import sklearn
from sklearn.model_selection import train_test_split#划分数据集
from sklearn.metrics import classification_report#分类报告
from sklearn.metrics import confusion_matrix#混淆矩阵

import keras
from keras.preprocessing.image import ImageDataGenerator#图片生成器
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,TensorBoard,EarlyStopping#回调函数返回学习速率；在每个训练期之后保存模型
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam,RMSprop
from keras.utils import np_utils#可视化


map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson', 
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak', 
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}

pic_size = 64#设定图片大小
batch_size = 128
epochs = 200
num_classes = len(map_characters)
pictures_per_class = 1000
test_size = 0.15

def load_pictures(BGR):
    """
    Load pictures from folders for characters from the map_characters dict and create a numpy dataset and 
    a numpy labels set. Pictures are re-sized into picture_size square.
    :param BGR: boolean to use true color for the picture (RGB instead of BGR for plt)
    :return: dataset, labels set
    """
    pics = []
    labels = []
    for k, char in map_characters.items():
        pictures = [k for k in glob.glob('./characters/%s/*' % char)]#从每类人物的文件夹里返回所有图片名字pictures=[****]
        #print(pictures)        
        #从pictures中选样本集，如果样本数目<pictures数目，则返回样本数目；如果大于，则返回pictures数目
        nb_pic = round(pictures_per_class/(1-test_size)) if round(pictures_per_class/(1-test_size))<len(pictures) else len(pictures)
        # nb_pic = len(pictures)
        for pic in np.random.choice(pictures, nb_pic):#从每类pictures中随机选np_pic张图片作为样本数据集
            a = cv2.imread(pic)#读取图片，默认彩色图,a.shape(x,x,3)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)#色彩空间转换BGR转为RGB
            a = cv2.resize(a, (pic_size,pic_size))#按比例缩放为pic_size * pic_size大小，此时a.shape(64,64,3)
            pics.append(a)
            labels.append(k)
    return np.array(pics), np.array(labels) 
def get_dataset(save=False, load=False, BGR=False):
    """
    Create the actual dataset split into train and test, pictures content is as float32 and
    normalized (/255.). The dataset could be saved or loaded from h5 files.
    :param save: saving or not the created dataset
    :param load: loading or not the dataset
    :param BGR: boolean to use true color for the picture (RGB instead of BGR for plt)
    :return: X_train, X_test, y_train, y_test (numpy arrays)
    """
    if load:
        h5f = h5py.File('dataset.h5','r')
        X_train = h5f['X_train'][:]
        X_test = h5f['X_test'][:]
        h5f.close()    

        h5f = h5py.File('labels.h5','r')
        y_train = h5f['y_train'][:]
        y_test = h5f['y_test'][:]
        h5f.close()
        
    else:
        X, y = load_pictures(BGR)#读取并获得图片信息
        #print(X.shape,y.shape)
        y = keras.utils.to_categorical(y, num_classes)#转换为one-hot编码
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)#拆分数据集
        if save:
            h5f = h5py.File('dataset.h5', 'w')
            h5f.create_dataset('X_train', data=X_train)
            h5f.create_dataset('X_test', data=X_test)
            h5f.close()

            h5f = h5py.File('labels.h5', 'w')
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('y_test', data=y_test)
            h5f.close()
            
    X_train = X_train.astype('float32') / 255.#归一化
    X_test = X_test.astype('float32') / 255.
    print("Train", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)
#    把每类的训练集和测试集数目打印出来
    if not load:
        dist = {k:tuple(d[k] for d in [dict(Counter(np.where(y_train==1)[1])), dict(Counter(np.where(y_test==1)[1]))]) 
                for k in range(num_classes)}
        print('\n'.join(["%s : %d train pictures & %d test pictures" % (map_characters[k], v[0], v[1]) 
            for k,v in sorted(dist.items(), key=lambda x:x[1][0], reverse=True)]))
    return X_train, X_test, y_train, y_test

def create_model_four_conv(input_shape):
    """
    CNN Keras model with 4 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    return model, opt

def create_model_six_conv(input_shape):
    """
    CNN Keras model with 8 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))#padding=same 输出与原始图像长度相同
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    '''
    model.add(Conv2D(512, (3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    '''
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    '''
    model.add(Dense(512))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    
    #model.add(Dense(256))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    '''
    model.add(Dense(num_classes, activation='softmax'))
    #opt = RMSprop(lr=0.0001, decay=1e-6)
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)#nesterov=True使用动量
    return model, opt

def create_model_vgg16(input_shape):
    """
    CNN Keras model with 6 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))#padding=same 输出与原始图像长度相同
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Conv2D(512, (3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Conv2D(512, (3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
  
    model.add(Dense(256))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    
    
    model.add(Dense(num_classes, activation='softmax'))
    #opt = RMSprop(lr=0.0001, decay=1e-6)
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)#nesterov=True使用动量
    return model, opt

def load_model_from_checkpoint(weights_path, six_conv=False, input_shape=(pic_size,pic_size,3)):
    if six_conv:
        model, opt = create_model_six_conv(input_shape)
    else:
        model, opt = create_model_four_conv(input_shape)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    return model,opt

#设置学习率衰减
def lr_schedule(epoch):
    initial_lrate = 0.03#初始学习率
    drop = 0.5#衰减为原来的多少倍
    epochs_drop = 12.0#每隔多久改变学习率
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))#math.pow(x,y)=x的y次方，math.floor向下取整
    #return lrate if lrate >= 0.0001 else 0.0001
    return lrate

def training(model, X_train, X_test, y_train, y_test, data_augmentation=True):
    """
    Training.
    :param model: Keras sequential model
    :param data_augmentation: boolean for data_augmentation (default:True)
    :param callback: boolean for saving model checkpoints and get the best saved model
    :param six_conv: boolean for using the 6 convs model (default:False, so 4 convs)
    :return: model and epochs history (acc, loss, val_acc, val_loss for every epoch)
    """
    if data_augmentation:
        #数据增强
        datagen = ImageDataGenerator(
            featurewise_center=False,  # 将输入数据的均值设置为 0，逐特征进行
            samplewise_center=False,  # 将每个样本的均值设置为 0
            featurewise_std_normalization=False,  # 将输入除以数据标准差，逐特征进行
            samplewise_std_normalization=False,  # 将每个输入除以其标准差
            zca_whitening=False,  # 应用 ZCA 白化
            rotation_range=10,  # 随机旋转的度数范围(degrees, 0 to 180)，旋转角度
            width_shift_range=0.1,  # 随机水平移动的范围，比例
            height_shift_range=0.1,  # 随机垂直移动的范围，比例
            horizontal_flip=True,  # 随机水平翻转，相当于镜像
            vertical_flip=False)  # 随机垂直翻转，相当于镜像
        
        # 根据一组样本数据，计算与数据相关转换有关的内部数据统计信息,当且仅当 featurewise_center 或 featurewise_std_normalization 或 zca_whitening 时才需要
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)
        
        ###每当val_cc有提升就保存checkpoint
        #save_best_only=True被监测数据的最佳模型就不会被覆盖，mode='max'保存的是准确率最大值
        filepath="weights_6conv_%s.hdf5" % time.strftime("%Y%m%d") 
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')    
        
        # 生成日志以便借助 TensorBoard 进行可视化分析。
        #tensorboard = TensorBoard(log_dir='logs', histogram_freq=0)
        #自动调节学习率
        #lrate = LearningRateScheduler(lr_schedule,verbose=1)
        
        #EarlyStopping
        #early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='min')
        
        callbacks_list = [checkpoint]
        history = model.fit_generator(datagen.flow(X_train, y_train,#传入 Numpy 数据和标签数组，生成批次的 增益的/标准化的 数据。在生成的批次数据上无限制地无限次循环。
                                    batch_size=batch_size),
                                    steps_per_epoch=X_train.shape[0] // batch_size,
                                    epochs=epochs,
                                    validation_data=(X_test, y_test),
                                    verbose=1,
                                    callbacks=callbacks_list)#调用一些列回调函数
        

        #查看分类报告，返回每类的精确率，召回率，F1值
        #P=TP/(TP+FP),R=TP/(TP+FN),F1=2PR/(P+R)
        score = model.evaluate(X_test,y_test,verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        y_pred = model.predict(X_test)
        print('\n', sklearn.metrics.classification_report(np.where(y_test > 0)[1], #y_test真实分类，np.where返
                                                          #回（array[],array[]），其中后面的array就是行方向上，y_test>0(1)的索引
                                                          np.argmax(y_pred, axis=1),#返回行方向上最大数值的索引
                                                          target_names=list(map_characters.values())), sep='') 
    
        #acc和loss可视化
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        #画出混淆矩阵
        plt.figure(figsize = (10,10))
        cnf_matrix = sklearn.metrics.confusion_matrix(np.where(y_test > 0)[1],np.argmax(y_pred, axis=1))
        classes = list(map_characters.values())
        thresh = cnf_matrix.max() / 2.#阈值
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, cnf_matrix[i, j],#在图形中添加文本注释
                     horizontalalignment="center",#水平对齐
                     color="white" if cnf_matrix[i, j] > thresh else "black")
        plt.imshow(cnf_matrix,interpolation='nearest',cmap=plt.cm.Blues)#cmap颜色图谱，默认RGB(A)
        plt.colorbar()#显示颜色条
        plt.title('confusion_matrix')#标题
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks,classes,rotation=90)
        plt.yticks(tick_marks,classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        #cnf_matrix = sklearn.metrics.confusion_matrix(np.where(y_test > 0)[1],np.argmax(y_pred, axis=1))
        #classes = list(map_characters.values())
        #plot_confusion_matrix(cnf_matrix,classes)
        
        
    else:
        history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,#用作验证集的训练数据的比例
          verbose=1,
          shuffle=True)#是否在每轮迭代之前进行数据混洗
        score = model.evaluate(X_test, y_test, verbose=1)
        
        #查看分类报告，返回每类的精确率，召回率，F1值
        #P=TP/(TP+FP),R=TP/(TP+FN),F1=2PR/(P+R)
        score = model.evaluate(X_test,y_test,verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        y_pred = model.predict(X_test)
        print('\n', sklearn.metrics.classification_report(np.where(y_test > 0)[1], #y_test真实分类，np.where返
                                                          #回（array[],array[]），其中后面的array就是行方向上，y_test>0(1)的索引
                                                          np.argmax(y_pred, axis=1),#返回行方向上最大数值的索引
                                                          target_names=list(map_characters.values())), sep='') 

        #acc和loss可视化
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        #画出混淆矩阵
        plt.figure(figsize = (10,10))
        cnf_matrix = sklearn.metrics.confusion_matrix(np.where(y_test > 0)[1],np.argmax(y_pred, axis=1))
        classes = list(map_characters.values())
        thresh = cnf_matrix.max() / 2.#阈值
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, cnf_matrix[i, j],#在图形中添加文本注释
                     horizontalalignment="center",#水平对齐
                     color="white" if cnf_matrix[i, j] > thresh else "black")
        plt.imshow(cnf_matrix,interpolation='nearest',cmap=plt.cm.Blues)#cmap颜色图谱，默认RGB(A)
        plt.colorbar()#显示颜色条
        plt.title('confusion_matrix')#标题
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks,classes,rotation=90)
        plt.yticks(tick_marks,classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    return model, history

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_dataset(load=True)
    model, opt = create_model_six_conv(X_train.shape[1:])
    #model, opt = create_model_vgg16(X_train.shape[1:])
    #model,opt = load_model_from_checkpoint('weights_6conv_20180602.hdf5', six_conv=True, input_shape=(pic_size,pic_size,3))
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    model, history = training(model, X_train, X_test, y_train, y_test, data_augmentation=True)
    
    
