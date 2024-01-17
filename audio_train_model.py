import  tensorflow as tf
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def extract_mfcc(audio_path):
    audio, _ = librosa.load(audio_path, sr=None, mono=True)
    mfccs = librosa.feature.mfcc(audio, sr=_, n_mfcc=13)
    return mfccs


def load_dataset(data_folder):
    X = []
    y = []

    label_names = ["real", "fake"]

    for label in label_names:
        label_path = os.path.join(data_folder, label)

        if os.path.exists(label_path) and os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                if filename.endswith('.wav'):
                    audio_path = os.path.join(label_path, filename)
                    mfccs = extract_mfcc(audio_path)
                    X.append(mfccs)
                    y.append(label)

    return X, y


data_folder = "D:\\DATASET\\ASVspoof2021_DF_eval_part00\\ASVspoof2021_DF_eval"
X, y = load_dataset(data_folder)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

X_train, X_test = np.array(X_train), np.array(X_test)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

input_shape = (X_train.shape[1], X_train.shape[2], 1)
model = create_model(input_shape)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

model.save('audio_deepfake_detection_model.h5')
