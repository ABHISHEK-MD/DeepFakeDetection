import os
from flask import Flask, render_template, request
import cv2
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

app = Flask(__name__)

model = load_model("trained_model.h5")

UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def file_save():
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)

    if ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower()) == 'mp4':
        return predict_video(file.filename)

    elif ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower()) == 'jpg':
        return images(file.filename)

def images(inp):
    path = UPLOAD_FOLDER + inp
    img = image.load_img(path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    predictions = model.predict(img_array)

    predicted_class = int(predictions[0] > 0.5)
    path_img = "static/uploads/" + inp

    if predicted_class == 1:
        result = "The Image is Predicted to be DeepFake."
    else:
        result = "The Image is Predicted to be Real."

    return render_template("predict_image.html", res=result, path=path_img)

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    predictions = []
    
    video_writer = cv2.VideoWriter(os.path.join(UPLOAD_FOLDER + "/frames", 'output_video.avi'),
                                   cv2.VideoWriter_fourcc(*'XVID'), fps,
                                   (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.resize(frame, (299, 299))
        frame = frame / 255.0
        cv2.imwrite(UPLOAD_FOLDER + "/frames/frame.jpg", frame)

        img_array = np.expand_dims(frame, axis=0)
        predictions.append(model.predict(img_array))

    cap.release()
    final_prediction = int(np.mean(predictions) > 0.5)
    
    if final_prediction == 1:
        result = "The video is predicted to be a deepfake."
    else:
        result = "The video is predicted to be real."

    path_main = "/static/uploads/" + video_path
    
    return render_template("predict_video.html", path=path_main, final=final_prediction)

def audio():
    return "Audio"

if __name__ == '__main__':
    app.run(debug=True)
