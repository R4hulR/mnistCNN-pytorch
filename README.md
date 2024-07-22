# Handwritten Digit Recognition with PyTorch

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset. The model achieves high accuracy in classifying handwritten digits.

## Project Structure

- `cnn_data/`: Directory to store the downloaded MNIST dataset (created automatically when running the notebook)
- `num6.png`: Sample image for testing the trained model (can be replaced with any single-digit image)
- `CNN-Pytorch.ipynb`: Jupyter Notebook containing the complete code for the project

## Dependencies

- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- matplotlib
- PIL

## How to Run

1. Install the required dependencies: ```pip install torch torchvision numpy pandas scikit-learn matplotlib pillow```
2. Open the Jupyter Notebook (`CNN-Pytorch.ipynb`) in Google Colab or your local environment.
3. Execute the code cells sequentially to:
- Download the MNIST dataset
- Train the CNN model
- Evaluate its performance
- Test it with a sample image

## Results

The trained model achieves a high accuracy of 98.59% on the MNIST test set, demonstrating its effectiveness in classifying handwritten digits.

## Future Work

- Experiment with different CNN architectures and hyperparameters
- Implement data augmentation techniques to increase training data diversity
- Deploy the trained model as a web application or API for real-time digit recognition

Contributions to this project are welcome. Feel free to open issues or submit pull requests.
