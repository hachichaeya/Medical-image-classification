 ###  Medical Image Classification using Vision Transformer### 
This project aims to classify benign tumors from medical images, specifically CT scans, using image processing techniques and the Vision Transformer model.

## Dataset
The dataset used for this project consists of CT scan images of patients with bone metastases. The dataset is divided into two classes: benign tumors and malignant tumors. It contains a total of 2000 images, with 1000 images in each class.

## Preprocessing
Before training the Vision Transformer model, the images are preprocessed using various image processing techniques. This includes resizing the images to a fixed size, applying contrast enhancement, and normalizing the pixel values.

## Vision Transformer Model
The Vision Transformer model is implemented using PyTorch. It consists of a series of Transformer blocks, which utilize self-attention mechanisms to capture long-range dependencies in the images. The output of the final Transformer block is fed into a fully connected layer, which produces the final classification output.

## Training
The Vision Transformer model is trained using a supervised learning approach. The dataset is split into training and validation sets, with 80% of the data used for training and 20% used for validation. The model is trained using a cross-entropy loss function and optimized using stochastic gradient descent.

## Evaluation
The trained Vision Transformer model is evaluated on a separate test set, which consists of unseen CT scan images. The evaluation metrics used include accuracy, precision, recall, and F1 score. The model's performance is compared to other traditional machine learning models and deep learning models to assess its effectiveness.

## Results
The results of the evaluation show that the Vision Transformer model achieves high accuracy and outperforms other models in classifying benign tumors from CT scan images. This indicates the potential of using Vision Transformer for medical image classification tasks.

## Conclusion
In conclusion, this project demonstrates the effectiveness of using the Vision Transformer model for classifying benign tumors from medical images, specifically CT scans. The model shows promising results and has the potential to improve the accuracy and efficiency of tumor detection in the medical field.

How to Run
Clone the repository
Install the necessary dependencies by running pip install -r requirements.txt
Run python preprocess.py to preprocess the dataset
Run python train.py to train the Vision Transformer model
Run python evaluate.py to evaluate the trained model on the test set
