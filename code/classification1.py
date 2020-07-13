'''采用自组CNN模型进行猫狗图片二分类'''
import os, shutil
import tensorflow as tf
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import math

# 自动分配显存，并且限制在50%内
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, )
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, )
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.compat.v1.Session(config=config)


# 原始数据集的路径
original_dataset_dir = '/home/Howie/alldataset/catsdogsdataset'

# 项目数据集存放路径
base_dir = '/home/Howie/kerasproject/cats_dogs/dataset'

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
fnames = ['cat.{}.jpg'.format(i) for i in range(2500)]
for fname in fnames:
    scr = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(scr, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(2500, 3000)]
for fname in fnames:
    scr = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(scr, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(3000, 3500)]
for fname in fnames:
    scr = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(scr, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(2500)]
for fname in fnames:
    scr = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(scr, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(2500, 3000)]
for fname in fnames:
    scr = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(scr, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(3000, 3500)]
for fname in fnames:
    scr = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(scr, dst)



# 建立CNN模型
def generate_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5,5), activation='relu',
    input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (5,5), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))


    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()
    return model

def cosine(base_lr, epoch):
    return base_lr * (math.cos((epoch%20) * math.pi/20) + 1 ) / (epoch//20 + 1)




# 图像生成器
train_datagen = ImageDataGenerator(rescale=1./255)

'''train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,)'''

validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, 
 target_size=(150,150), batch_size=256, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir,
 target_size=(150,150), batch_size=256, class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

model = generate_model() 
acc, val_acc, loss, val_loss = [], [], [], []


model.compile(optimizer=optimizers.RMSprop(lr=5e-5), 
loss='binary_crossentropy', metrics=['acc'])
# 训练模型
history_bin = model.fit(train_generator, 
epochs=120, validation_data=validation_generator)

    
acc = acc + history_bin.history['acc'] 
val_acc = val_acc + history_bin.history['val_acc'] 
loss = loss + history_bin.history['loss']
val_loss = val_loss + history_bin.history['val_loss'] 


 #保存模型
model.save('/home/Howie/kerasproject/cats_dogs/cats_and_dogs_small_3.h5')

# 训练过程可视化
epoch = range(1, len(acc)+1)



plt.subplot(211)
plt.plot(epoch, acc, 'b', label='Training accuracy')
plt.plot(epoch, val_acc, 'y', label='Validation_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

plt.subplot(212)
plt.plot(epoch, loss, 'b', label='Training loss')
plt.plot(epoch, val_loss, 'y', label='Validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()


plt.show()
plt.savefig("/home/Howie/kerasproject/cats_dogs/cats_and_dogs_small_3.png")






