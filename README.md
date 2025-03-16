# 📧 Spam Mail Detection Using Logistic Regression

## 📌 Project Overview
This project implements a **Spam Mail Detection** system using **Logistic Regression**. The model classifies emails as either **Spam (0)** or **Ham (1)** based on their content using **TF-IDF Vectorization** for feature extraction.

## 🚀 Features
- Load and preprocess email dataset
- Convert text data into numerical features using **TF-IDF Vectorization**
- Train a **Logistic Regression** model
- Evaluate model performance using **accuracy score**
- Predict if a new email is spam or ham

## 📂 Dataset
The dataset contains email messages labeled as:
- **Spam (0):** Unwanted or junk emails
- **Ham (1):** Legitimate emails

## 🔧 Technologies Used
- **Python**
- **Pandas, NumPy** (Data Processing)
- **Scikit-learn** (Machine Learning - Logistic Regression, TF-IDF, Accuracy Score)

## 🏗️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-mail-detection.git
   cd spam-mail-detection
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model training script:
   ```bash
   python train.py
   ```

## 📊 Model Training Process
1. **Load Dataset:**
   ```python
   df = pd.read_csv('//enter your System path')
   ```
2. **Preprocess Data:**
   - Handle missing values
   - Encode labels (Spam = 0, Ham = 1)
   ```python
 
3. **Split Data:**
   
4. **Feature Extraction (TF-IDF):**
   ```python
   
5. **Train Logistic Regression Model:**
   ```python
  
6. **Evaluate Model Performance:**
   ```python
   
7. **Predict Email as Spam or Ham:**
   ```python
   

## 📜 Results
- Achieved **high accuracy** using **Logistic Regression**
- Model effectively classifies spam emails with precision

## 🔗 References
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Kaggle Spam Mail Dataset](https://www.kaggle.com/datasets)

## 🤝 Contributing
Contributions are welcome! Feel free to open a pull request.

## 📜 License
This project is licensed under the MIT License.

