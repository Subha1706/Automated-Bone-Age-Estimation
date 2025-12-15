# 🦴 Automated Bone Age Estimation using Machine Learning & Deep Learning

## 📌 Project Overview
This project focuses on **automated bone age estimation** from hand radiograph images using a combination of **classical machine learning models**, **transfer learning**, and a **custom Convolutional Neural Network (CNN)**.

The objective is to evaluate multiple modeling paradigms and identify the most effective approach for accurate, robust, and scalable bone age prediction under real-world clinical conditions.

This work is developed as part of a **PRML Course Project**, with a **research paper currently in progress**.

---

## 🎯 Objectives
- Automate bone age prediction from hand X-ray images  
- Compare classical ML, EfficientNet-B0, and custom CNN approaches  
- Study the impact of preprocessing and segmentation on performance  
- Minimize prediction error while ensuring model stability and interpretability  

---

## 🧠 Approaches Implemented

### 1️⃣ Classical Machine Learning Models
Handcrafted radiomic features are extracted from segmented images and used for regression.

**Features Extracted**
- Intensity statistics (mean, standard deviation, quantiles)  
- Edge features (Canny edge density)  
- Gradient features (Sobel magnitude statistics)  
- Texture features (LBP, GLCM)  
- Shape features (area, perimeter, eccentricity)  

**Models Used**
- Random Forest Regressor  
- LightGBM Regressor  
- Ensemble Model (RF + LightGBM + CatBoost)  

**Best Classical Result**
- MAE ≈ 17.39 months (Ensemble)

---

### 2️⃣ Transfer Learning — EfficientNet-B0
- Pretrained on ImageNet  
- Fine-tuned with a regression head  
- Input size: 224 × 224 × 3  
- Optimizer: Adam  
- Loss function: MAE / MSE  

**Outcome**
- Faster convergence than classical models  
- Better feature representation  
- Slightly lower performance compared to the custom CNN  

---

### 3️⃣ Custom CNN (Final Selected Model)
A task-specific CNN designed for medical image regression.

**Architecture Highlights**
- Convolution → BatchNorm → ReLU → MaxPool blocks  
- Increasing filter sizes: 32 → 64 → 128 → 256  
- Dropout layers to reduce overfitting  
- Dense layers: 512 → 128  
- Single neuron regression output  

**Final Performance**
- MAE: 9.46 months  
- RMSE: 12.24 months  
- R² Score: 0.91  

✔️ Selected as the best-performing model

---

## 🧪 Image Preprocessing & Segmentation

### Preprocessing
- Grayscale conversion  
- CLAHE (Contrast Limited Adaptive Histogram Equalization)  
- Normalization and resizing  

### Segmentation
Two segmentation strategies were evaluated:
- Approach A: Otsu + Canny + Watershed (discarded due to instability)  
- Approach B: Morphological operations + contour-based ROI extraction (selected)

Segmentation significantly improved model stability and accuracy.

---

## 📊 Evaluation Metrics
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² Score  
- Confusion matrix for age-group classification  
- Gender bias analysis  

---

## 📁 Repository Structure
```bash
├── notebooks/
│   ├── rsna-grp-10-approach1-2.ipynb
│
├── reports/
│   └── PRML_Project_Report.pdf
│
├── figures/
│   ├── preprocessing_examples.png
│   ├── segmentation_results.png
│
├── README.md
```


---

## 🚀 How to Run

### Clone the Repository
```bash
git clone https://github.com/your-username/automated-bone-age-estimation.git
cd automated-bone-age-estimation
```

### Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python tensorflow torch
```

### Run the Notebook
```bash
jupyter notebook notebooks/rsna-grp-10-approach1-2.ipynb
```

## Research Paper (In Progress)
A research paper is currently being prepared based on this project, focusing on:
- Comparative analysis of ML and DL approaches
- Impact of segmentation on prediction accuracy
- Error interpretation and clinical reliability
- Justification of CNN architecture choices
  
The paper will be linked here upon completion.

## Key Takeaways
- Deep learning models significantly outperform handcrafted feature-based methods
- Proper preprocessing and segmentation are critical for medical imaging tasks
- Task-specific CNNs can outperform large pretrained architectures
- Ensemble learning improves classical ML performance but lags behind DL models

## References
- RSNA Bone Age Dataset
- EfficientNet-B0 (ImageNet pretrained)
- CLAHE, Otsu Thresholding, Canny Edge Detection, Watershed Algorithm
- Scikit-learn, TensorFlow, PyTorch

## Author

**Subhasree Yenigalla**
B.Tech Computer Science and Engineering
