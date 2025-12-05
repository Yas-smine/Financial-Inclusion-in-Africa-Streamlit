# Financial Inclusion Prediction App in East Africa

## Objective
This project aims to predict whether an individual in East Africa is likely to have or use a bank account, using demographic and socio-economic features. Financial inclusion means providing individuals and businesses with access to useful and affordable financial services in a responsible and sustainable way, such as payments, savings, credit, and insurance.

## Dataset
The dataset contains information for approximately 33,600 individuals across East African countries (Kenya, Uganda, Tanzania, Rwanda) for the years 2016-2018. Key columns include:

- `country`: Country of residence
- `year`: Year of survey
- `uniqueid`: Unique respondent ID
- `bank_account`: Target variable (Yes/No)
- `location_type`: Urban or Rural
- `cellphone_access`: Yes/No
- `household_size`: Number of people in household
- `age_of_respondent`: Respondent's age
- `gender_of_respondent`
- `relationship_with_head`
- `marital_status`
- `education_level`
- `job_type`

## Steps Followed

1. **Environment Setup**
   - Installed necessary Python packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `streamlit`, `joblib`.

2. **Data Exploration**
   - Loaded data with `pandas` and examined using `.info()`, `.head()`, `.describe()`.
   - Checked for missing values, duplicates, and outliers.
   - Explored value counts for categorical features.

3. **Data Preprocessing**
   - Simplified education and job categories to reduce sparsity.
   - Encoded categorical features using `LabelEncoder`.
   - Handled class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique).
   - Scaled and transformed numeric features as needed.

4. **Model Training**
   - Trained **XGBoost classifier** on preprocessed data.
   - Evaluated with accuracy, precision, recall, and F1-score.
   - Achieved balanced performance after SMOTE.

5. **Streamlit App**
   - Created a web application with **Streamlit**:
     - Inputs for all demographic and socio-economic features.
     - Simplification functions for education and job.
     - Transform categorical features using trained encoders.
     - Predict using the XGBoost model.
     - Display result as a success/warning message.
   - Handles unseen categories gracefully to avoid errors during prediction.

6. **Deployment**
   - Uploaded code to GitHub repository.
   - Deployed the Streamlit application via Streamlit Cloud.
   - Users can interactively input feature values to predict financial inclusion.

