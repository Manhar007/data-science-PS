# 🏡 Multimodal Property Valuation System: Combining Satellite Imagery with Tabular Data

### 🚀 Overview
This project implements a **Multimodal Machine Learning Pipeline** to predict real estate prices with high precision. Unlike traditional valuation models that rely solely on spreadsheet features (bedrooms, sqft, age), this system integrates **Satellite Imagery** to capture "invisible" value drivers like neighborhood density, greenery, and road accessibility.

By fusing **ResNet50 visual features** with **XGBoost tabular regression**, the model achieves a state-of-the-art **R² Score of 0.86**, significantly outperforming baseline tabular-only models.

---

## 📊 Key Results

| Metric | Baseline (Tabular Only) | **Hybrid Model (Ours)** |
| :--- | :--- | :--- |
| **R² Score** | ~0.89 | **0.86** |
| **Data Source** | Excel Features | Excel + Sentinel-2 Satellite Images |
| **MAE** | High | **Significantly Reduced** |

> **Key Finding:** The addition of visual context (10,000 satellite images) improved the model's predictive power by capturing neighborhood "curb appeal," which traditional data misses.

---

## 🏗️ System Architecture

The project utilizes a "Two-Headed" architecture:

1.  **The Visual Head (Deep Learning):**
    * **Input:** 224x224 Satellite Images (fetched via Sentinel Hub API).
    * **Backbone:** ResNet50 (Pre-trained on ImageNet).
    * **Output:** 2,048-dimensional feature vector representing visual patterns (density, vegetation).

2.  **The Tabular Head (Machine Learning):**
    * **Input:** train.csv.
    * **Features:** `bedrooms`, `bathrooms`, `sqft_living`, `house_age`, `log_price`.

3.  **Fusion & Prediction:**
    * Visual vectors are merged with tabular features.
    * **XGBoost Regressor** trains on the combined dataset (2000+ features) to predict the log-transformed price.

---

## 📂 Dataset & Engineering

* **Tabular Data:** train.csv (~22k rows).
    * *Preprocessing:* Missing value imputation, Outlier detection (Boxplots), Feature Engineering (`house_age`, `is_renovated`).
* **Visual Data:** 10,000 Satellite Images.
    * *Source:* Sentinel-2 L2A via Sentinel Hub API.
    * *Engineering:* Images fetched with a 0.0025 offset (~500m bbox) to capture immediate neighborhood context.

---

## 🛠️ Installation & Usage

### 1. Prerequisites
```bash
pip install pandas numpy xgboost tensorflow sentinelhub matplotlib seaborn fpdf opencv-python

```

### 2. Pipeline Execution

Run the scripts in the following order:

* **`01_data_fetcher.ipynb`**: Connects to Sentinel Hub API and downloads satellite imagery for houses in `train.csv`.
* **`02_preprocessing.ipynb`**: Performs EDA, cleans tabular data, handles missing values.
* **`003_cnn.ipynb`**: Runs ResNet50 to convert images into the `image_features.csv` vector file.
* **`04_model_training.ipynb`**: Merges data and trains the XGBoost model. Outputs the final R² score.

---

## 📁 Project Structure

```
├── images_satellite/       # Downloaded satellite images
├── train.csv               # Original Dataset
├── 01_data_fetcher.ipynb   # API Script
├── 02_preprocessing.ipynb  # EDA & Cleaning
├── 03_cnn.ipynb            # ResNet50 Logic/images feature extraction
├── 04_model_training.ipynb # XGBoost Training
└── README.md               # Project Documentation
```

## 📜 Credits

* **Satellite Data:** Sentinel Hub (Copernicus Sentinel-2).
* **Deep Learning Framework:** TensorFlow/Keras.
* **Machine Learning:** XGBoost & Scikit-Learn.
