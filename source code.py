# import all required tools and libraries
from tensorflow.python import keras
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dropout, Dense, Flatten 
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#load mnist dataset
train_data, test_data= mnist.load_data()

#preprocess train data
X, Y = train_data[0], train_data[1]  # X - pixels, Y-labels
img_rows, img_cols=28, 28     # image size 28X28 pixels
num_img = len(X)
num_classes = 10  # 10 labels 0-9
Y = to_categorical(Y, num_classes)

# cross validation of train data
X = X.reshape(num_img, img_rows, img_cols, 1)/255
x_train,  x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.1, random_state = 1 )

# preprocess test data
X_test = test_data[0]
num_test_img = len(X_test)
x_test = X_test.reshape(num_test_img, img_rows, img_cols, 1)/255
y_test = test_data[1]
y_test = to_categorical(y_test, num_classes)


#preprocess train data with ImageDataGenerator
datagen = ImageDataGenerator(rotation_range = 10,  
                             zoom_range = 0.10,  
                             width_shift_range = 0.1, 
                             height_shift_range = 0.1)

# create, compile and fit model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), input_shape=(img_rows, img_cols, 1), padding='Same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(5,5), padding='Same', strides=2))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(32, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(5,5), padding='Same', strides=2))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, y_train, batch_size=10), epochs=30, validation_data=(x_val, y_val),
                    callbacks=[EarlyStopping(patience=3, restore_best_weights=True),
                    ReduceLROnPlateau(monitor = "val_loss", factor = 0.01, patience = 3,
                    verbose = 0, mode = "auto", epsilon = 1e-04, cooldown = 0,
                    min_lr = 0)])

#evaluate model with test data
eval=model.evaluate(x_test, y_test)
print(eval)

# approx. accuracy 0.994

