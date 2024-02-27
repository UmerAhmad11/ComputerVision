import cv2 as cv
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

# Define constants
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_EPOCHS = 50

# Define data generators for train and test sets with data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'photos/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    'photos/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Define your CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')  # Assuming you have 7 mood classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define learning rate scheduler
def lr_scheduler(epoch):
    lr = 0.001
    if epoch > 30:
        lr *= 0.5
    elif epoch > 20:
        lr *= 0.75
    elif epoch > 10:
        lr *= 0.9
    return lr

lr_callback = LearningRateScheduler(lr_scheduler)

# Train the model
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // BATCH_SIZE,
          epochs=NUM_EPOCHS,
          validation_data=test_generator,
          validation_steps=test_generator.samples // BATCH_SIZE,
          callbacks=[lr_callback])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print('Test accuracy:', test_acc)

# Save the model
model.save('ml_improved.h5')

