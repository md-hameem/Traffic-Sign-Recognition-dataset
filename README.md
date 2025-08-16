# Traffic Sign Recognition Using Deep Learning

This project uses **Convolutional Neural Networks (CNN)** to recognize traffic signs from images. The model is trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset and is capable of classifying traffic signs in real-time from live video feeds.

## Project Overview

In this project, we build and train a deep learning model to recognize and classify traffic signs. The dataset consists of 43 different classes of traffic signs, and the task is to create a robust model that can predict the correct class for each input image.

### Key Features:
- **Image Classification**: Predict traffic sign classes from images.
- **Real-Time Prediction**: Model can classify traffic signs from a live video stream.
- **Data Augmentation**: To enhance model generalization, we applied data augmentation techniques such as rotation, translation, and flipping.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites
- Python 3.7+
- TensorFlow (2.x)
- OpenCV
- scikit-learn
- numpy
- matplotlib
- pillow

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic-sign-recognition.git
   cd traffic-sign-recognition
2. Create a virtual environment and activate it (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   .\venv\Scripts\activate   # For Windows
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model, simply run the following script:

```bash
python train_model.py
```

This will load the dataset, apply data augmentation, and train the model on the traffic sign classification task. The trained model will be saved in the `models/` directory.

### Real-Time Prediction

To perform real-time predictions using your webcam or a video file, run:

```bash
python predict.py
```

This will capture frames from your webcam or video file, preprocess the frames, and use the trained model to classify traffic signs in real-time.

### Model Evaluation

To evaluate the performance of the trained model on the test data, you can run:

```bash
python evaluate_model.py
```

This will load the test data, evaluate the model, and print the accuracy.

## Model Architecture

The model is a **Convolutional Neural Network (CNN)** designed to classify traffic signs. The architecture is as follows:

1. **Convolutional Layers**: These layers are responsible for feature extraction from the images.
2. **MaxPooling Layers**: To reduce the spatial dimensions of the feature maps.
3. **Fully Connected Layers**: These layers flatten the 3D feature maps into 1D and make predictions.
4. **Softmax Output Layer**: A 43-class softmax output layer to classify images into one of 43 traffic sign classes.

The model was trained using the **Adam optimizer** with categorical cross-entropy loss.

## Dataset

The dataset used in this project is the **German Traffic Sign Recognition Benchmark (GTSRB)**, which consists of over **50,000 labeled images** of traffic signs.

### Dataset Details:

* **Classes**: 43 different traffic sign classes (e.g., stop signs, speed limits, yield signs).
* **Resolution**: Images were resized to **64x64 pixels** for model training.
* **Data Splits**: The dataset is split into training, validation, and test sets.

You can download the dataset from Kaggle using the following command:

```bash
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
```

Extract the dataset and place it in the `data/` directory.

## Results

### Model Performance

After training the model, we achieved an accuracy of **{your\_accuracy}%** on the test set. This model can classify traffic signs with a high degree of accuracy, making it suitable for real-time applications.


## Contributing

We welcome contributions! If you find any issues or have ideas for improvements, feel free to create an issue or submit a pull request.

### How to Contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

