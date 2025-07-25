## DeepFake Detection

### Overview
This project demonstrates a web application for image,video and audio classification using a deep learning model. The model is trained to classify images as either real or manipulated (deepfake).

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

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
### Training the model
1. Prepare your dataset:

    Organize your dataset with two classes - real and fake images. Place them in separate directories inside a 'data' directory.

    ```
    data/
    ├── real/
    │   ├── real_image1.jpg
    │   ├── real_image2.jpg
    │   └── ...
    ├── fake/
    │   ├── fake_image1.jpg
    │   ├── fake_image2.jpg
    │   └── ...
    ```

2. Train the model:

    ```bash
    python train_model.py --train_data_dir path/to/train_data --validation_data_dir path/to/validation_data
    ```

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
