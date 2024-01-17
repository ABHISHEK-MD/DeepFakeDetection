# RJPOLICE_HACK_494_t0b3h4ck3r5_8

## Deepfake Image Detection

### Overview
This project demonstrates a web application for image classification using a deep learning model. The model is trained to classify images as either real or manipulated (deepfake).

### Requirements
- Python (version 3.x)
- TensorFlow
- Keras
- Flask

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/deepfake-detection.git
    cd deepfake-detection
    ```

### Training the model
1. Prepare your dataset:

    Organize your dataset with two classes - real and fake images. Place them in separate directories inside a 'data' directory.

2. Train the model:

    This will train the deepfake detection model using the provided dataset.
   
### Web Application
1. Run the Flask web application:

    ```bash
    python app.py
    ```

    This will start the Flask server. Open your browser and go to http://127.0.0.1:5000/ to use the web application.
   
### Usage
- Upload an image through the web interface, and the model will predict whether it's a deepfake or not.

### View the results:
The system will generate output indicating the likelihood of the input image containing deepfake content.
