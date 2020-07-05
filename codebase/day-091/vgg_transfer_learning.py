import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


IMAGE_SIZE = [150,150]

epochs = 10
batch_size = 64

train_path = "dogs-cats/train/"
validation_path = "dogs-cats/train/"

image_files = glob(train_path+"/*/*.jp*g")

validation_files = [image_file for image_file in image_files if 'dog' in image_file][:2500]
validation_files.extend([image_file for image_file in image_files if 'cat' in image_file][:2500])


folders = glob(train_path+"/*")

vgg = VGG16(input_shape=IMAGE_SIZE + [3],weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
prediction = Dense(len(folders),activation='softmax')(x)

model = Model(inputs=vgg.input,outputs=prediction)
print (model.summary())

model.compile(
        loss="categorical_crossentropy",
        optimizer='rmsprop',
        metrics=['accuracy']
        )

gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=preprocess_input
        )

test_gen = gen.flow_from_directory(validation_path,target_size=IMAGE_SIZE)
print (test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k,v in test_gen.class_indices.items():
    labels[v] = k



train_generator = gen.flow_from_directory(
            train_path,
            target_size=IMAGE_SIZE,
            shuffle=True,
            batch_size=batch_size
        )

valid_generator = gen.flow_from_directory(
            validation_path,
            target_size=IMAGE_SIZE,
            shuffle=True,
            batch_size=batch_size
        )

r = model.fit_generator(
            train_generator,
            #validation_data=valid_generator,
            epochs=15,
            steps_per_epoch=len(image_files)//batch_size,
            #validation_steps=len(validation_files)//batch_size
        )

def get_confusion_matrix(data_path,N):
    print ("Generating Confusion Matrix {}".format(N))
    predictions = []
    targets = []
    i = 0
    for x,y in gen.flow_from_directory(data_path,target_size=IMAGE_SIZE,shuffle=False,batch_size=batch_size*2):
        i += 1
        if i%50 ==0: 
            print (i)
        p = model.predict(x)
        p = np.argmax(p,axis=1)
        y = np.argmax(y,axis=1)
        predictions = np.concatenate((predictions,p))
        targets = np.concatenate((targets,y))
        if len(targets) >= N:
            break
    cm = confusion_matrix(targets,predictions)
    return cm

cm = get_confusion_matrix(train_path,len(image_files))
print (cm)


