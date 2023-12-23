import os
from flask import Flask, render_template, request, jsonify
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model('your_model.h5')

# Define the path for uploading images
UPLOAD_FOLDER = 'A:/RAJASTHAN/IMageDetection/static/images/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed extensions for uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image upload and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)

        # Load and preprocess the image for prediction
        img = image.load_img(img_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.  # Normalize the image

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted class
        predicted_class = int(predictions[0] > 0.5)

        # Return the result
        if predicted_class == 1:
            result = "The image is predicted to be a deepfake."
        else:
            result = "The image is predicted to be real."

        return jsonify({'result': result})

    else:
        return jsonify({'error': 'File type not allowed'})

if __name__ == '__main__':
    app.run(debug=True)
