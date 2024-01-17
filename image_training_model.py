from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras import layers
from keras import models
from keras import optimizers

train_data_dir = 'A:/RAJASTHAN/IMageDetection/Train'
validation_data_dir = 'A:/RAJASTHAN/IMageDetection/Validation'


batch_size = 32
img_size = (299, 299)
epochs = 10

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'  
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)


base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))


model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification

base_model.trainable = False

model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)
    
model.save('image_model.h5')
