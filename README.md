# Fashion MNIST Classification Project

This project was developed as part of an advanced machine learning course, demonstrating feature extraction techniques and model optimization.
The objective was to achieve the best classification accuracy while optimizing computational efficiency.

## Table of Contents  
1. [Introduction](#-introduction)  
2. [Dataset Information](#dataset-information)  
3. [Data Preprocessing](#data-preprocessing)  
   - [3.1 Checking for Missing Values](#31-checking-for-missing-values)  
   - [3.2 Class Distribution Analysis](#32-class-distribution-analysis)  
   - [3.3 Normalization (Pixel Scaling)](#33-normalization-pixel-scaling)  
   - [3.4 Reshaping Data for Machine Learning Models](#34-reshaping-data-for-machine-learning-models)  
   - [3.5 Converting NumPy Arrays to Pandas DataFrame](#35-converting-numpy-arrays-to-pandas-dataframe)  
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
   - [4.1 Summary Statistics for Pixel Intensities](#41-summary-statistics-for-pixel-intensities)  
   - [4.2 Brightness & Variability Across Categories](#42-brightness--variability-across-categories)  
   - [4.3 Class Similarity Analysis (Cosine Similarity)](#43-class-similarity-analysis-cosine-similarity)  
   - [4.4 Pixel Intensity Distribution Across Clothing Categories](#44-pixel-intensity-distribution-across-clothing-categories)  
   - [4.5 Identifying the Most Important Pixels Using Correlation](#45-identifying-the-most-important-pixels-using-correlation)  
   - [4.6 Class-Wise Pixel Correlation Heatmap](#46-class-wise-pixel-correlation-heatmap)  
   - [4.7 Pixel Variance Analysis](#47-pixel-variance-analysis)  
5. [Feature Engineering & Dimensionality Reduction](#feature-engineering--dimensionality-reduction)  
   - [5.1 Principal Component Analysis (PCA)](#51-principal-component-analysis-pca)  
   - [5.2 Independent Component Analysis (ICA)](#52-independent-component-analysis-ica)  
   - [5.3 PCA & ICA Combined Workflow for Feature Selection](#53-pca--ica-combined-workflow-for-feature-selection)  
6. [Machine Learning Models Used](#machine-learning-models-used)  
7. [Model Evaluation & Performance](#model-evaluation--performance)  
8. [Key Findings & Conclusions](#key-findings--conclusions)  
9. [Future Improvements](#future-improvements)  
10. [Contact](#contact)  

---

## **1. Introduction**  
This project focuses on classifying the **Fashion MNIST dataset** using various **machine learning algorithms**, including **Ensemble Learning techniques**.  
The study explores the impact of **feature transformation techniques (PCA, ICA)**, hyperparameter tuning, and data augmentation on model performance.  

The main objectives of this project are:  
- To compare different **machine learning models** and identify the most effective approach.  
- To analyze the impact of **feature engineering & dimensionality reduction** on classification accuracy.  
- To evaluate misclassification trends and propose improvements.  

---

## **2. Dataset Information**  
The Fashion MNIST dataset contains **70,000 grayscale images (28x28 pixels each)** categorized into **10 classes** representing different clothing types.  

### **Dataset Composition**  
- **Training Set:** 60,000 images  
- **Test Set:** 10,000 images  
- **Image Dimensions:** 28x28 pixels (784 features per sample)  
- **Classes:**  
  1. T-shirt/top  
  2. Trouser  
  3. Pullover  
  4. Dress  
  5. Coat  
  6. Sandal  
  7. Shirt  
  8. Sneaker  
  9. Bag  
  10. Ankle boot  

---

## **3. Data Preprocessing**  

### **3.1 Checking for Missing Values**  
- Verified dataset integrity; no missing values were found.  

### **3.2 Class Distribution Analysis**  
- Ensured class balance to avoid bias in training.  

### **3.3 Normalization (Pixel Scaling)**  
- Scaled pixel values to range **[0,1]** to enhance numerical stability.  

### **3.4 Reshaping Data for Machine Learning Models**  
- Flattened images from **28x28 matrices to 784-dimensional vectors** for classifier input.  

### **3.5 Converting NumPy Arrays to Pandas DataFrame**  
- Converted structured data for easier analysis and manipulation.  

---

## **4. Exploratory Data Analysis (EDA)**  

### **4.1 Summary Statistics for Pixel Intensities**  
- Analyzed the mean and standard deviation of pixel values across categories.  

### **4.2 Brightness & Variability Across Categories**  
- Examined intensity variations across different clothing items.  

### **4.3 Class Similarity Analysis (Cosine Similarity)**  
- Measured how visually similar some classes are to each other.  

### **4.4 Pixel Intensity Distribution Across Clothing Categories**  
- Plotted pixel intensity histograms to observe category-wise differences.  

### **4.5 Identifying the Most Important Pixels Using Correlation**  
- Evaluated which pixels contribute most to classification accuracy.  

### **4.6 Class-Wise Pixel Correlation Heatmap**  
- Visualized pixel dependencies using correlation matrices.  

### **4.7 Pixel Variance Analysis**  
- Determined which pixels contain the most variation across samples.  

---

## **5. Feature Engineering & Dimensionality Reduction**  

### **5.1 Principal Component Analysis (PCA)**  
- Reduced **784 features to 43**, preserving **85% of variance**.  

### **5.2 Independent Component Analysis (ICA)**  
- Extracted independent components after PCA for further feature refinement.  

### **5.3 PCA & ICA Combined Workflow for Feature Selection**  
- Implemented a **hybrid PCA → ICA approach** to maximize feature interpretability while minimizing computational cost.  

---

## **6. Machine Learning Models Used**  
The following models were evaluated for classification performance:  

| Model | Description |
|-----------------|----------------------------------------------|
| Logistic Regression | Baseline linear classifier |
| K-Nearest Neighbors (KNN) | Distance-based classification |
| Naive Bayes | Probabilistic model assuming feature independence |
| Random Forest | Ensemble learning with multiple decision trees |
| XGBoost | Gradient boosting algorithm for optimized classification |
| AdaBoost | Adaptive boosting technique for weak learners |

---

## **7. Model Evaluation & Performance**  

| Model | Accuracy (%) |
|---------------------|----------------|
| **XGBoost** | **87.93%** |
| **KNN** | **86.68%** |
| **AdaBoost** | **86.07%** |
| **Random Forest** | **84.67%** |
| **Logistic Regression** | **82.38%** |
| **Naive Bayes (Dummy)** | **74.57%** |

---

## **8. Key Findings & Conclusions**  
- **PCA effectively reduced dimensionality, improving efficiency without sacrificing accuracy.**  
- **ICA contributed to faster computation but had minimal impact on classification accuracy.**  
- **Random Forest and XGBoost achieved the highest accuracy.**  
- **Data augmentation helped reduce misclassification, especially for visually similar categories.**  

---

## **9. Future Improvements**  
- Implement Convolutional Neural Networks (CNNs) for better feature extraction.  
- Explore **advanced augmentation techniques** (e.g., GAN-based synthetic image generation).  

---

## **10. Contact**  
For any questions, reach out:  
- **Email:** tomgoz96@gmail.com  
- **GitHub:** [Tomer Gozlan](https://github.com/tomergozlan)  
