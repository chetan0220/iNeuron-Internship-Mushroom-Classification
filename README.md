# Mushroom Classification Project

This repository contains the code and resources for the Mushroom Classification project, developed during the ineuron.ai internship. The project focuses on building and deploying a machine learning model to classify mushrooms as edible or poisonous based on various features. The final model is an ensemble of multiple machine learning models, deployed using Streamlit.

[Live Website Link](https://mushroom-classifier.streamlit.app/)

## Dataset

The dataset used for this project was sourced from Kaggle. It contains various attributes of mushrooms which are used to predict whether a mushroom is edible or poisonous.

## Proposed System

The proposed system involves the following steps:
1. **Data Collection**: Collect the mushroom dataset from Kaggle.
2. **Data Transformation**: Transform the data into a suitable format for model training.
3. **Data Preprocessing**: Perform preprocessing tasks such as handling missing values, encoding categorical variables, and scaling features.
4. **Machine Learning Models**: Train various machine learning models including Logistic Regression, K-Nearest Neighbor, Naive Bayes, Support Vector Machine, Decision Tree, and Artificial Neural Network.
5. **Ensemble Model**: Combine the predictions from individual models to create an ensemble model for improved accuracy.
6. **Streamlit Cloud Setup**: Set up Streamlit for model deployment.
7. **Deployment**: Deploy the ensemble model using Streamlit for web-based interaction.

## Accuracy Graph for All Models

The accuracy of the various models used in the project is compared in the graph below:

![Model Accuracy Comparison](https://github.com/user-attachments/assets/e8e8c15e-d548-4a5d-8dfd-73297e292ff1)

## Website

The deployed model can be accessed and used via the following Streamlit web application:

[Mushroom Classifier Web App](https://mushroom-classifier.streamlit.app/)

## Conclusion

This project demonstrates the effective use of ensemble learning to improve the accuracy of mushroom classification. By combining multiple models, the final ensemble model achieves high accuracy, providing reliable predictions on the edibility of mushrooms. The deployment on Streamlit ensures that the model is easily accessible for end-users, making it a practical tool for mushroom classification.

The repository includes all necessary code, datasets, and resources to replicate and extend the project. Feel free to explore and contribute to enhance the functionality and performance of the Mushroom Classification system.
