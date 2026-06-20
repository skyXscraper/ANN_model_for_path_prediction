# ANN Model for Path Prediction

A Python-based project that implements an **Artificial Neural Network (ANN)** to predict path outcomes using data extracted from CPF files. This repository contains scripts for data parsing, preprocessing, model training, testing, and generating output predictions.

---

## Project Overview

This repository is designed to train and evaluate an ANN that predicts paths (or outcomes) based on input data extracted from CPF (Common Path Format) and related files. The model is implemented in Python using standard machine-learning libraries.

---

## Features

-  **Data Extraction & Parsing**  
  Scripts to parse CPF data files and extract relevant features.

-  **Preprocessing**  
  Clean and prepare data for model training.

-  **ANN Training**  
  Multiple training scripts to build and train neural network models for path prediction.

-  **Output Prediction**  
  Generate and save results for evaluation or downstream processing.

---


---

##  Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package manager)

---

###  Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/skyXscraper/ANN_model_for_path_prediction.git
    cd ANN_model_for_path_prediction
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

##  Usage

### 1. Preprocess the Data

Prepare and clean the raw CPF-extracted data:

```bash
python preprocessing.py
```

### 2. Train the ANN Model

You can choose between different ANN training scripts based on configuration or experimentation needs.
Each script may use different network architectures or training parameters.

### 3. Test / Evaluate the Model

Run the test script to evaluate the trained model and generate predictions.
python test.py


## Dependencies

All required Python libraries are listed in requirements.txt.
Common dependencies include:

1. numpy
2. pandas
3. scikit-learn
4. tensorflow / keras (if used in ANN scripts)
5. Other supporting ML and data-processing libraries

pip install -r requirements.txt

