import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import os

# Load the trained model
model = load_model('traffic_sign_model.keras')

# Path to test data
dataset_path = '/kaggle/input/gtsrb-german-traffic-sign/Test'
img_size = 64

def load_data(data_path):
    images = []
    labels = []
    for class_label in os.listdir(data_path):
        class_folder = os.path.join(data_path, class_label)
        if os.path.isdir(class_folder):
            for img_file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(class_label)

    images = np.array(images)
    labels = np.array(labels)

    images = images / 255.0  # Normalize pixel values

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels, num_classes=43)

    return images, labels

# Load the test data
X_test, y_test = load_data(dataset_path)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
