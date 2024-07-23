
# WCE Curated Colon Disease Classification using Deep Learning

This project focuses on building a machine learning model to classify images related to colon diseases using a Convolutional Neural Network (CNN) based on the VGG16 architecture. The application is deployed using Flask for easy user interaction through a web interface.

## Table of Contents:
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Web Application](#web-application)
- [Screenshots](#screenshots)
- [Contributors](#contributors)

## Installation

1. Clone the Repository:
```
git clone https://github.com/Srinathu0/WCE-Curated-Colon-Disease-Classification-using-Deep-Learning
cd CURATED_COLON-DISEASE
```
2. Create a Virtual Environment:
```
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install the Required Packages:
```
pip install -r requirements.txt
```
## TensorFlow Version Notice

To ensure compatibility between the model training and deployment environments, please make sure to use TensorFlow version 2.16.2 in both environments.

```bash
pip install tensorflow==2.16.2
```

## Usage
### Training the Model
1. Open and run the Jupyter Notebook `WCE.ipynb` to train the model:
- Ensure the dataset is properly loaded.
- Run the cells to build, and train the model.
2. Save the trained model:
```
model.save('cnn.h5')
```
## Dataset
- The dataset consists of images related to different colon diseases, categorized into four classes: Normal, Ulcerative, Polyps, and Esophagitis.
- The images are resized to 224x224 pixels for input to the VGG16 model.
- Follow the link to access the Dataset: https://www.kaggle.com/datasets/francismon/curated-colon-dataset-for-deep-learning

## Model Training
- The VGG16 model is used as a base with pre-trained weights from ImageNet.
- The model's layers are frozen, and custom layers are added for classification.
- The model is trained using train_data and validated using test_data.
- Open the Jupyter Notebook: 
```
jupyter notebook WCE.ipynb
```

## Model Evaluation
- The model's accuracy and loss are plotted for both training and validation datasets.
- The evaluation can be done using history object from the training process.

## Web Application
- The web application uses Flask to allow users to upload images and receive predictions on the type of colon disease.
- The uploaded image is preprocessed and passed to the trained model for prediction.

## Running the Web Application
- Ensure the trained model cnn.h5 is in the project directory.
- Run the Flask application:
```
python app.py
```

## Screenshots
![image](https://github.com/user-attachments/assets/dd2a0701-4a95-43c5-ab2a-0a5444694fc3)
![image](https://github.com/user-attachments/assets/612b036b-5384-4dad-80fa-d02db21889ad)
![image](https://github.com/user-attachments/assets/3e14cd75-b095-40b0-800f-a1bcaa8d26d1)
![image](https://github.com/user-attachments/assets/656c5bad-e7ab-447e-b379-cf1e80ccde03)

## 🔗 Contributors :
- Srinath U:  [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/srinath-u-26ba0b226)
- Vanshika Agrawal:  [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vanshika-agrawal-908b3b2aa)
- Homasri Mandyam: [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/homasri-mandyam-2a804a314)
- Prashant Choudhary:  [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/prashant-choudhary-b05a15265)

