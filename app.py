import os
from flask import *
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, jsonify,redirect
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import librosa
app = Flask(__name__)

model = load_model("D:\\police\\RJPOLICE_HACK_494_t0b3h4ck3r5_8-main\\trained_model.h5")


UPLOAD_FOLDER = "static\\uploads\\"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','mp4','wav','mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def file_save():
    file = request.files['file']
    if file and allowed_file(file.filename):
        img_path = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        file.save(img_path)

    if ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower()) == 'mp4':
        return predict_video(file.filename)


    elif ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower()) == 'jpg':
         return images(file.filename)
    elif ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower()) == 'wav':
         return predict_audio(file.filename)

def images(inp):
        path = UPLOAD_FOLDER+inp
        img = image.load_img(path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        predictions = model.predict(img_array)

        predicted_class = int(predictions[0] > 0.5)
        path_img = "static/uploads/"+inp


        if predicted_class == 1:
            result = "The Image is Predicted to be DeepFake."
        else:
            result = "The Image is Predicted to be Real."

        return render_template("predict_image.html",res=result,path = path_img)
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    predictions = []
    video_writer = cv2.VideoWriter(os.path.join(UPLOAD_FOLDER+"/frames", 'output_video.avi'),
                                   cv2.VideoWriter_fourcc(*'XVID'), fps,
                                   (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.resize(frame, (299, 299))  
        frame = frame / 255.0
        cv2.imwrite(UPLOAD_FOLDER+"/frames",frame)

        img_array = np.expand_dims(frame, axis=0)
        predictions.append(model.predict(img_array))

    cap.release()
    final_prediction = int(np.mean(predictions) > 0.5)
    if final_prediction == 1:
        result = "The video is predicted to be a deepfake."
    else:
        result = "The video is predicted to be real."
    print(result)
    path_main = "/static/uploads/"+video_path
    print(path_main)
    return render_template("predict_video.html",path = path_main,res = result,conditional=True)

def predict_audio(inp):
    audio_path = UPLOAD_FOLDER+inp
    max_len = 174
    model = tf.keras.models.load_model("C:\\Users\\Bhavaneeshwaran\\Downloads\\Telegram Desktop\\audio_classification_model.h5")
    def extract_mfcc(audio_path, max_len=174):
        audio, sr = librosa.load(audio_path, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        pad_width = max(0, max_len - mfccs.shape[1])
        mfccs_padded = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        return mfccs_padded

    def predict_audio(model, audio_path, max_len=174):
        mfccs = extract_mfcc(audio_path, max_len)
        mfccs = np.expand_dims(mfccs, axis=-1)
        mfccs = np.expand_dims(mfccs, axis=0)

        prediction = model.predict(mfccs)
        predicted_class = np.argmax(prediction)

        return predicted_class

    predicted_class = predict_audio(model, audio_path, max_len)

    def predict_audio_probabilities(model, audio_path, max_len=174):
        mfccs = extract_mfcc(audio_path, max_len)
        mfccs = np.expand_dims(mfccs, axis=-1)
        mfccs = np.expand_dims(mfccs, axis=0)

        probabilities = model.predict(mfccs)

        return probabilities

    predicted_probabilities = predict_audio_probabilities(model, audio_path, max_len)
    print(predicted_class)
    if predicted_class == 1:
        return render_template("predict_audio.html",res = "The Audio is Deepfake")
    else:
        return render_template("predict_audio.html",res = "The Audio is Real")
if __name__ == '__main__':
    app.run(debug=True)
