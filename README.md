# Simpson-Recognition


### 此文是在学习机器学习过程中对实践项目的一点总结，若有错漏之处，请指正，敬请谅解

使用工具为Python3，Keras,本文更偏重基础并增加了实现过程中的心得体会，例如代码的注释与讲解，调参策略等，对于想拿此项目练手的同学更具有指导性，同样本文将从以下几个方面进行介绍：

- **数据预处理**
- **构建模型**
- **模型训练**
- **模型评估**
- **模型优化**
- **超参数选择**
- **调参策略**


数据集下载链接:https://pan.baidu.com/s/14hUSlqipz8yTWGdgJCXkaw  密码:hrmc，目前我这里的人物类别有47类，项目选取了其中18类作为样本数据集，样本图片大小不一，样式千奇百怪，背景也不尽相同，随便贴几张图大家感受下



看完后有什么感想？是不是觉得想要达到100%的识别准确率是真的难- -|||

### 1.数据预处理

从文件夹中选取样本，每个人物选取的训练集样本占比为0.85，1000个样本，测试集占比0.15，如果选取的图片数目小于该人物的总图片数，则从中随机选取，否则选取该人物所有的图片作为样本数据集，然后通过OpenCV来读取图片，因为OpenCV默认通道为BGR，所以需要对图片转换为咱们熟悉的RGB图像

```
a = cv2.imread(pic)
a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
```

### 注意，对于深度学习来说，输入的图片大小必须是一致的，因为只有这样才会统一编码，所以每张图片必须resize成统一大小

`a = cv2.resize(a, (pic_size,pic_size))`

然后将读取的label转为one-hot编码

`y = keras.utils.to_categorical(y, num_classes)`

完成上述步骤后，将选取的样本save下来，下次直接load就可以，否则调参失去意义，最后也是非常重要的数据归一化

```
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
    pics = []
    labels = []
    for k, char in map_characters.items():
        pictures = [k for k in glob.glob('./characters/%s/*' % char)]#从每类人物的文件夹里返回所有图片名字
        #print(pictures)        
        #从pictures中选样本集，如果样本数目<pictures数目，则返回样本数目；如果大于，则返回pictures数目
        nb_pic = round(pictures_per_class/(1-test_size)) if round(pictures_per_class/(1-test_size))<len(pictures) else len(pictures)
        # nb_pic = len(pictures)
        for pic in np.random.choice(pictures, nb_pic):#从每类pictures中随机选np_pic张图片作为样本数据集
            a = cv2.imread(pic)#读取图片
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)#色彩空间BGR转为RGB
            a = cv2.resize(a, (pic_size,pic_size))#按比例缩放为pic_size * pic_size大小
            labels.append(k)
    return np.array(pics), np.array(labels) 

def get_dataset(save=False, load=False, BGR=False):
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
###把每类的训练集和测试集数目打印出来
    if not load:
        dist = {k:tuple(d[k] for d in [dict(Counter(np.where(y_train==1)[1])), dict(Counter(np.where(y_test==1)[1]))]) 
                for k in range(num_classes)}
        print('\n'.join(["%s : %d train pictures & %d test pictures" % (map_characters[k], v[0], v[1]) 
            for k,v in sorted(dist.items(), key=lambda x:x[1][0], reverse=True)]))
    return X_train, X_test, y_train, y_test
```

### 2.构建模型

构建卷积神经网络，带有6个ReLU激活函数的卷积层及4个池化层和3个全连接层，之所以构造6个卷积层，首先是因为辛普森一家里的人物确实长得都太像了，稍不留神就成了脸盲，其次是人物的形态不一，人脸分布在不同的区域，所以要更加细致的提取特征，以达到区分的能力。卷积池化层还增加了Dropout，可以有效防止过拟合，输出层采用softmax函数来输出各类概率，优化器optimizer选用随机梯度下降SGD，学习率lr=0.01，学习率衰减decay=1e-6，动量momentum=0.9，动量是为了越过平坦区域或泥石流区域(左右震荡)。

```
def create_model_six_conv(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))#padding=same 输出与原始图像大小相同
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)#nesterov=True使用动量
    return model, opt
```

### 3.模型训练

先对原始数据集进行训练，epochs为200，通常情况下，头几次训练epochs尽可能的大，以便观察模型训练情况。另外我还用到回调函数(callbaks)中的ModelCheckpoint，实现的功能是将val_acc最高的模型保存下来，然后测试时直接载入使用。callbaks概念摘自官方中文文档，回调函数是一个函数的合集，会在训练的阶段中所使用。你可以使用回调函数来查看训练模型的内在状态和统计。你可以传递一个列表的回调函数（作为 callbacks 关键字参数）到 Sequential 或 Model 类型的 .fit() 方法。在训练时，相应的回调函数的方法就会被在各自的阶段被调用。说白了就是在训练时想干点别的，可以通过callbaks来完成。

```
###每当val_acc有提升就保存checkpoint
#save_best_only=True被监测数据的最佳模型就不会被覆盖，mode='max'保存的是准确率最大值
filepath="weights_8conv_%s.hdf5" % time.strftime("%Y%m%d") 
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,#用作验证集的训练数据的比例
          verbose=1,#日志显示进度条
          shuffle=True,#是否在每轮迭代之前进行数据混洗
          callbacks=checkpoint)
score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


Train (14317, 64, 64, 3) (14317, 18)
Test (2527, 64, 64, 3) (2527, 18)
Train on 12885 samples, validate on 1432 samples
Epoch 1/200
12885/12885 [==============================] - 4s 323us/step - loss: 2.8252 - acc: 0.0869 - val_loss: 2.7111 - val_acc: 0.1648
Epoch 2/200
12885/12885 [==============================] - 3s 261us/step - loss: 2.5845 - acc: 0.1918 - val_loss: 2.4444 - val_acc: 0.2479
...
Epoch 199/200
12885/12885 [==============================] - 3s 265us/step - loss: 0.0063 - acc: 0.9981 - val_loss: 0.5669 - val_acc: 0.9148
Epoch 200/200
12885/12885 [==============================] - 3s 264us/step - loss: 0.0050 - acc: 0.9983 - val_loss: 0.5910 - val_acc: 0.9225
2527/2527 [==============================] - 0s 138us/step
Test loss: 0.5129188157600574
Test accuracy: 0.9283735651624594
```

分别画出训练集和测试集上的accuracy和loss变化曲线，注意，模型训练完后返回的是一个History对象。其History.history属性是连续 epoch 训练损失和评估值，以及验证集损失和评估值的记录。

```
#acc和loss可视化
# accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```


### 4.模型评估

验证集的Test accuracy为0.928，感觉还不错，但是根据上面的曲线可以看出，网络发生过拟合，再看看分类报告

```
#查看分类报告，返回每类的精确率，召回率，F1值
#P=TP/(TP+FP),R=TP/(TP+FN),F1=2PR/(P+R)
score = model.evaluate(X_test,y_test,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
y_pred = model.predict(X_test)
print('\n', sklearn.metrics.classification_report(np.where(y_test > 0)[1], 
                                                  np.argmax(y_pred, axis=1),
                                                  target_names=list(map_characters.values())), sep='')

                            precision   recall  f1-score   support

  abraham_grampa_simpson       0.95      0.90      0.93       167
  apu_nahasapeemapetilon       0.96      0.99      0.97        73
            bart_simpson       0.89      0.91      0.90       161
charles_montgomery_burns       0.94      0.90      0.92       165
            chief_wiggum       0.95      0.96      0.95       165
          comic_book_guy       0.93      0.92      0.93        74
          edna_krabappel       0.92      0.89      0.90        73
           homer_simpson       0.89      0.86      0.87       174
           kent_brockman       0.91      0.97      0.94        87
        krusty_the_clown       0.92      0.97      0.94       170
            lisa_simpson       0.91      0.85      0.88       178
           marge_simpson       0.98      0.99      0.98       169
     milhouse_van_houten       0.97      0.99      0.98       145
             moe_szyslak       0.93      0.88      0.90       172
            ned_flanders       0.95      0.95      0.95       184
            nelson_muntz       0.88      0.83      0.85        42
       principal_skinner       0.93      0.95      0.94       197
            sideshow_bob       0.89      0.97      0.93       131

             avg / total       0.93      0.93      0.93      2527
```

画出混淆矩阵

```
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
```

可以看出，Lisa的准确率偏低，很大部分误分成Bart，因为什么呢？先看看他俩长啥样再说，你会发现，他俩除了头型其他确实很像！这下分错也情有可原了- -||


### 5.模型优化

接着分析过拟合发生的原因，可能是训练样本数目太少，也可能是模型复杂等，那么就分别改进方法来验证什么原因导致，先从模型复杂度入手，可能Dropout力度不够，改为0.5试试，跑完测试结果如下：

```
Test loss: 0.31535795541368883
Test accuracy: 0.9271863867135495
```


可以看出，模型仍然过拟合，验证集准确率不但没有提升反而下降，并且Lisa的准确率仍然很低，基本可以排除是模型复杂导致的过拟合，那么很有可能是样本数量太少导致，接下来试下数据增强(data augmentation)，这里用到Keras的ImageDataGenerator

```
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
```

这里主要针对图像处理，主要用到rotation_range(人物歪着头)，width_shift_range(脸部不完整，请看文章开头的图片)，height_shift_range，horizontal_flip(脸部朝左朝右)四个功能，当然其他的你也可以自己尝试，然后训练，flow()返回一个生成器，用来扩充数据，每次生成batch_size个样本

```
history = model.fit_generator(datagen.flow(X_train, y_train,
                                    batch_size=batch_size),
                                    steps_per_epoch=X_train.shape[0] // batch_size,
                                    epochs=epochs,
                                    validation_data=(X_test, y_test),
                                    verbose=1,
                                    callbacks=callbacks_list)#调用一些列回调函数

Test loss: 0.10107234056333621
Test accuracy: 0.9790265136761396
```

由图看出，过拟合问题解决了，根源就在于数据集量太少，而且准确率提升很明显！所以针对小数据集，data augmentation真的很重要！到这里模型训练结果基本上不会有较大的提升了，剩下的就是继续不断地尝试，对模型优化，调参，让结果无限接近100%。

### 6.超参数选择

超参数选取的好，可以给模型训练起到锦上添花的作用，当然深度学习还是要“唯结果论”。

- **可变学习率(LearningRateScheduler)**

```
def lr_schedule(epoch):
    initial_lrate = 0.03#初始学习率
    drop = 0.5#衰减为原来的多少倍
    epochs_drop = 12.0#每隔多久改变学习率
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(lr_schedule,verbose=1)
```

这个实现的功能跟本文前面用到的学习率衰减(lr decay)类似，在优化的过程中，学习率应该不断的减小，保证在山坡上大步迈，接近山谷时小步走，我写了一个函数，模型每训练12次将学习率减半。但是训练结果并没有多少提升，可能是因为没调到最优参数。

- **EarlyStopping**

```
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
```

实现的功能是提前停止训练，可以避免过拟合，当val_loss不再下降时，模型会自动停止训练。patience是设定val_loss没有下降的次数，超过这个次数则停止训练，开始的时候可以先不用EarlyStopping跑几次看看抖动的次数，然后再设定比该次数稍大就行。mode有‘auto’, ‘min’, ‘max’三种可能，根据monitor设定，如果是val_loss就设定min，val_acc设定max，要是不清楚该选哪个就选auto，不过要是知道的话还是建议设置下，确保没问题。

### 7.调参策略
- **刚开始可以先构建简单的网络结构模型，如果是大数据集可以先选取小量样本训练，如果泛化能力表现可以，再训练更深更大的网络**

- **卷积层激活函数一般选用ReLU，全连接输出层一般用softmax**

- **出现过拟合时，如果数据集小，那么可以尝试采用data augmentation，如果数据集大，可以用Dropout来降低模型复杂度或者调节learning rate，采用lr decay或可变学习率根据结果取舍，并配合EarlyStopping提高效率**

- **尝试用RMSprop，Adam，Adadelta来优化函数，效果通常也不错，但我更喜欢SGD+momentum，因为操作可控且效果好**

- **采用Batch Normalization，绝对的神器，每一层输出进入激活函数前，将数据统一分布成均值为0，方差为1的标准正态分布，相当于将原本映射到饱和区域拉到中间区域，可以大大的提高收敛速度，epoch可以减小一半以上**

- **增加网络深度，同时要增加epoch，层数多提取特征多，模型复杂意味着训练时间更长**

- **调整卷积核数目，卷积核数目不宜过大，否则容易过拟合**

- **要更多关注val_loss，因为下一轮epoch的val_loss上升，val_acc也跟着提高的情况也不是没有，毕竟loss是优化目标**
