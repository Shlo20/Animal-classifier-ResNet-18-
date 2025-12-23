# Animal Classifier Confidence Study

This project analyzes the confidence behavior of a convolutional neural network
(ResNet-18) trained to classify animal images into 10 categories.

Rather than focusing only on accuracy, this study examines how confident the
model is in its predictions and whether those confidence scores match correctness.

## Dataset
The model was trained and evaluated on an animal image dataset with:
- 10 animal categories
- 26,179 total images
- 5,235 images used for evaluation

(The dataset is not included in this repository.)

## Methods
- Model: ResNet-18 (PyTorch)
- Confidence analysis using:
  - Confidence histograms
  - Beta distribution fitting
  - Reliability diagrams
  - Expected Calibration Error (ECE)

## Results
The model achieved very high classification accuracy, but confidence analysis
showed that predictions were often extremely confident, including some incorrect
predictions. This highlights the importance of calibration in machine learning
models.

## Technologies Used
- Python
- PyTorch
- NumPy
- pandas
- matplotlib

## Environment Notes
This project was developed and tested using PyTorch with CUDA 12.1 on an NVIDIA GPU

## How to Run
Install dependencies:
```bash
pip install -r requirements.txt
