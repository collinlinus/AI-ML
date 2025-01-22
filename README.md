**Introduction**  
The goal of this project is to leverage computer vision techniques to detect the faces of sea turtles in images. Deep learning models are used to identify the unique facial features which can help in population monitoring, tracking health and conserving endangered species.   
The model was trained using a datset containing sea turtle images with annotations highlighting the facial features for supervised learning. Convolutional Neural Networks (CNNs) were used to train the model to recognize and classify the sea turtle faces effectively.  



**Technologies Used**  
1. Python: Programming language used for data processing and model development.  
2. TensorFlow: This library is used for buidling and training the deep learning model.  
3. NumPy: For numerical operations on the data.   
4. Matplotlib/Seaborn: Libraries for data visualization and model evaluation.  
5. Google Coalb: For experimentation and testing purposes.



**Dataset**  
The dataset containing the images to be used for model deployment had each imge labeled with bounding boxes around the sea turtles faces. The dataset was sourced from kaggle.com and includes images of sea turtles in various environments such as oceans and beaches. 



**Model Training**  
The model used for sea turtle face detection is based on a Convolutional Neural Network (CNN) which invloves the following steps:

1. Pre-processing: Here images from the dataset are normalized, resized and augmented to improve the model's performance.
2. Model Architecture: The CNN is used to extract fetures and classify the presence of sea turtle faces in the images from the dataset.
3. Training: Model training is done using the labeled dataset, with the training and validation sets split to monitor the model's performance.
4. Fine-tuning/Alteration: In order to achive the best results, hyper-parameters such as batch size, epochs are used with this model.



**Model Evaluation**  
The performance of the model is evaluated using some of the following metrics:

1. Accuracy: This relates to the percentage of correct predictions made by the model.
2. Precision and Recall:  This ensures that the model correctly detects sea turtle faces without too many false positives/negatives. 
