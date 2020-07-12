'''采用自组CNN模型进行猫狗图片二分类'''
import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



# 原始数据集的路径
original_dataset_dir = '/home/Howie/alldataset/catsdogsdataset'

# 项目数据集存放路径
base_dir = '/home/Howie/kerasproject/catsdogs_classification/dataset'

# 创建训练集、验证集、测试集目录
train_dir = os.path.join(base_dir, 'train')
#os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
#os.mkdir(test_dir)

# 创建猫狗目录
train_cats_dir = os.path.join(train_dir, 'cats')
validation_cats_dir = os.path.join(validation_dir, 'cats')
test_cats_dir = os.path.join(test_dir, 'cats')
#os.mkdir(train_cats_dir)
#os.mkdir(validation_cats_dir)
#os.mkdir(test_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_dogs_dir = os.path.join(test_dir, 'dogs')
#os.mkdir(train_dogs_dir)
#os.mkdir(validation_dogs_dir)
#os.mkdir(test_dogs_dir)

#数据集复制
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    scr = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    #shutil.copyfile(scr, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    scr = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    #shutil.copyfile(scr, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    scr = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    #shutil.copyfile(scr, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    scr = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    #shutil.copyfile(scr, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    scr = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    #shutil.copyfile(scr, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    scr = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    #shutil.copyfile(scr, dst)



# 建立CNN模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu',
 input_shape=(150,150,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# 选择优化器
model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4),
 loss='binary_crossentropy', metrics=['acc'])

# 图像生成器
train_dategen = ImageDataGenerator(rescale=1./255)
validation_dategen = ImageDataGenerator(rescale=1./255)

train_generator = train_dategen.flow_from_directory(train_dir, 
 target_size=(150,150), batch_size=40, class_mode='binary')
validation_generator = validation_dategen.flow_from_directory(validation_dir,
 target_size=(150,150), batch_size=20, class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# 训练模型
history = model.fit_generator(train_generator, steps_per_epoch=50,
 epochs=2, validation_data=validation_generator, validation_steps=50)

 #保存模型
model.save('/home/Howie/kerasproject/catsdogs_classification/cats_and_dogs_small_1.h5')

# 训练过程可视化
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoch = range(1, len(acc)+1)


plt.plot(epoch, acc, 'b', label='Training accuracy')
plt.plot(epoch, val_acc, 'y', label='Validation_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.show()





