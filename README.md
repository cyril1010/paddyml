## Crop Guard - Deep Learning Model Using CNN and ResNet50

## Overview
This project implements a deep learning model using Convolutional Neural Networks (CNN) and ResNet50 architecture to classify different diseases in rice crops. The model is trained on a dataset containing images of rice leaves affected by various diseases like blast, blight, spot, and tungro.

## Functionality
- **Image Preprocessing**: Utilizes Keras for image preprocessing, including data augmentation and normalization.
- **Model Training**: Loads the pre-trained ResNet50 model and trains a new model on top of it, freezing the base layers and adding custom dense layers for classification.
- **Model Evaluation**: Evaluates the trained model on a test dataset to measure accuracy and loss.
- **Prediction**: Predicts the class of rice disease in both local images and images from online links.
- **Model Saving and Conversion**: Saves the trained model to Google Drive and converts it to TensorFlow Lite (.tflite) format for deployment on mobile devices.

## Usage
- **Access Dataset**: Mounts Google Drive to access the dataset containing training, testing, and validation images.
- **Load Trained Model**: Loads the pre-trained model from Google Drive for inference.
- **Image Preprocessing**: Preprocesses images using Keras ImageDataGenerator.
- **Train Model**: Trains the model using the pre-trained ResNet50 base and custom dense layers.
- **Test Image Links**: Provides links to sample images for testing the model.
- **Save Model**: Saves the trained model to Google Drive for future use.
- **Convert Model**: Converts the saved .h5 model to TensorFlow Lite (.tflite) format for deployment.

## Dependencies
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Requests
- PIL

## Getting Started
1. Ensure all necessary libraries are installed.
2. Mount Google Drive to access the dataset and save the trained model.
3. Load the pre-trained model or train a new one using the provided dataset.
4. Evaluate the model's performance and save it to Google Drive.
5. Convert the saved model to TensorFlow Lite format for deployment.

## License
This project is licensed under the MIT License.

## Acknowledgments
- This project utilizes the power of deep learning and transfer learning with the ResNet50 architecture.

## Contact
For any inquiries or suggestions, please contact:
- Email: kucyril7@gmail.com
- LinkedIn: https://www.linkedin.com/in/cyril1010

## Author
- Name: Cyril K U
