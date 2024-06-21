# CodSoft
# Machine Learning Projects

## SMS Spam Detection

### Overview
This project classifies SMS messages as spam or not spam using a Multinomial Naive Bayes classifier trained on TF-IDF vectorized data. It includes a Streamlit interface for user interaction.

### Technologies Used
- Python
- Libraries: numpy, pandas, re, sklearn, streamlit

### Features
- **Data Preprocessing**: Text cleaning, tokenization using regular expressions.
- **Model Training**: Upsampling of minority class (spam) to handle class imbalance, TF-IDF vectorization, Multinomial Naive Bayes classification.
- **Streamlit Interface**: Web application for message classification.

### Usage
1. **Installation**
   - Clone the repository:
     ```
     git clone <repository_url>
     ```
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```

2. **Running the Application**
   - Navigate to the project directory and run the Streamlit app:
     ```
     streamlit run sms_spam_detection.py
     ```
   - Enter a message to classify it as spam or not spam.

### Example
![image](https://github.com/MinaNabil730/CodSoft/assets/109760458/f0db447c-2726-4a80-84e5-d36476c72596)


---

## Credit Card Fraud Detection

### Overview
This project detects fraudulent credit card transactions using various machine learning models, focusing on preprocessing, balancing the dataset, and model evaluation.

### Technologies Used
- Python
- Libraries: pandas, seaborn, matplotlib, sklearn, xgboost

### Features
- **Data Acquisition**: Downloads dataset from Kaggle using Kaggle API.
- **Data Preprocessing**: Handling missing values, dropping duplicates, feature engineering (age from date of birth).
- **Model Training**: Comparison of multiple classifiers (Logistic Regression, Decision Tree, Random Forest, etc.) to detect fraud.
- **Model Evaluation**: Metrics include Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and Classification Report.

### Usage
1. **Downloading the Dataset**
   - **Using Kaggle API**: 
     - Sign up for a Kaggle account and generate an API token (`kaggle.json`).
     - Place `kaggle.json` in the `~/.kaggle` directory (create it if it doesn't exist).
     - Run the script to download and unzip the dataset:
       ```bash
       python download_dataset.py
       ```
   - **Manually**: Download the dataset from the following links:
     - [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
     - **Note**: If downloading manually, update the paths (`train_data_path` and `test_data_path`) in the script to your dataset location. The "Downloading the data online" cell does not need to be run.

2. **Installation and Running**
   - Clone the repository and navigate to the project directory.
      ```
     git clone <repository_url>
     ```
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```

3. **Running the Application**
   - Explore and run the Jupyter notebook or Python script (`fraud_detection.py`) to execute the fraud detection models.

---

### Dataset Links

- [SMS Spam Detection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

### Note for Manual Download
- If downloading the dataset manually, ensure to update the paths (`train_data_path` and `test_data_path`) in the script to your dataset location.
- The script assumes the dataset is located in the `data/` directory within the project structure.

---

### License
This project is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.

### Acknowledgments
These projects were completed as part of the requirements for the CODSoft Certificate.

