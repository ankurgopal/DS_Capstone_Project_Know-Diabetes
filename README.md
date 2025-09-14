Early Diabetes Detection -A Data Driven Approach for Early Intervention & Prevention
========================================================================================

📝 Project Overview:

This project utilizes Machine Learning techniques to improve the accuracy of predicting the risk and progression of Type 2 Diabetes by analyzing demographic, lifestyle, and clinical health indicators. By identifying high-risk individuals early and supporting more precise prognosis, this project aims to enable timely interventions and improve long-term health outcomes and quality of life for patients.

📖 Table of Contents

- Project Motivation
- Problem Statement
- Opportunities
- Impact
- Key Beneficiaries
- ML Models
- The Data
- Road Map
- Future Development
- Appendix

🎯  Project Motivation

Type 2 Diabetes runs in my family, so I’ve seen firsthand the long-term health challenges and financial strain it can cause. This project is driven by a personal and societal need to identify high-risk individuals earlier using Machine Learning — helping to prevent complications, reduce healthcare costs, and improve overall quality of life.


### Problem Statement:

This project will attempt to find a solution to the following:

"How can we use Machine Learning to enhance the early detection and prognosis of Type 2 Diabetes in at-risk individuals?"

🌟 Opportunities

- Early Risk Identification- By accurately identifying individuals at risk of developing Type 2 Diabetes, we can enable timely lifestyle and medical interventions that may prevent or delay disease onset.

- Personalized Preventive Care: Machine learning models can support personalized health recommendations, helping individuals make informed choices about diet, physical activity, and routine screenings.

- Reduced Healthcare Burden: Early detection and proactive care can reduce long-term complications such as cardiovascular disease, neuropathy, and kidney failure — ultimately lowering healthcare costs and improving patient outcomes.

- Support for Public Health Policy: Aggregated insights from this project can inform targeted public health strategies and resource allocation in high-risk populations.

- Integration into Clinical Tools: With robust validation, the model could be integrated into electronic health record systems to assist healthcare providers in making data-driven risk assessments during routine checkups. 

🌍 Impact

- Improved Patient Outcomes: Early identification of high-risk individuals enables timely intervention, reducing the likelihood of serious complications such as vision loss, heart disease, stroke, kidney failure, and limb amputation.

- Enhanced Quality of Life: Empowering individuals with awareness and preventive strategies helps them maintain healthier lifestyles, reducing the physical, emotional, and financial burdens of chronic disease.

- Informed Clinical Decisions: Healthcare providers can use predictive insights to tailor monitoring and treatment plans, improving the effectiveness of care and reducing trial-and-error in diagnosis or management.

- Cost Savings to the Healthcare System: Preventing or delaying the onset of Type 2 Diabetes and its complications significantly reduces long-term treatment costs for patients, hospitals, and governments.

- Scalable Public Health Benefits: Once validated, this model can be adapted across diverse populations and healthcare systems to aid in widespread diabetes prevention and policy planning.


👨‍👩‍👧‍👦 Key Beneficiaries

- Individuals who lead an unhealthy lifestyle. They could benefit from using these datasets and see what are the main factors which causes diabetes and what are the steps to take to avoid it 

- Health Professionals- Early detection or predicting high risk individuals who can have this disease would help professionals to prioritize healthcare. They could also help targeted campaigns and health education for such individuals. 

- Government- Insights from this data would help the government better spend their budget on healthcare and provide suitable resources.


🧠 Machine Learning Models

Machine Learning can offer scalable, data driven predictions on which individuals are most likely to be at risk for diabetes. Machine learning will help us identify which features are critical and whether there is a correlation and causation. 

I will be attempting to use below Predictive Models :

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- XGBoost

📈 The Data

This data was collected by using Behavioural Risk Factor Surveillance System (BRFSS) which is a health-related telephone survey collected by Centre for Disease Control and Prevention (CDC). 

I scrapped the data from UCI repository which is a well known & trusted source for data collection.

https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators


| **Feature Name**          | **Description**                                             |
| ------------------------- | ----------------------------------------------------------- |
| `Diabetes_Class`          | Target variable: 0 = No, 1 = Has diabetes                   |
| `HighBP`                  | 1 = High blood pressure, 0 = Normal                         |
| `HighChol`                | 1 = High cholesterol, 0 = Normal                            |
| `CholCheck`               | 1 = Had cholesterol checked in last 5 years, 0 = No         |
| `BMI`                     | Body Mass Index (weight in kg / height in m²)               |
| `Smoker`                  | 1 = Current/former smoker, 0 = Never smoked                 |
| `Stroke`                  | 1 = History of stroke, 0 = No                               |
| `HeartDisease_or_Attack`  | 1 = Heart attack or CHD history, 0 = No                     |
| `Physical_Activity`       | 1 = Had physical activity in past 30 days, 0 = No           |
| `Fruits`                  | 1 = Eats fruit ≥ once/day, 0 = No                           |
| `Veggies`                 | 1 = Eats vegetables ≥ once/day, 0 = No                      |
| `Heavy_Alcohol_Use`       | 1 = Heavy alcohol use, 0 = No                               |
| `AnyHealthcare`           | 1 = Has any form of health coverage, 0 = No                 |
| `Doctor_unavailable_cost` | 1 = Couldn’t see doctor due to cost, 0 = No                 |
| `General_Health`          | Self-reported health status: 1 = Excellent to 5 = Poor      |
| `Mental_Health`           | Number of days of poor mental health in the past 30 days    |
| `Physical_Health`         | Number of days of poor physical health in the past 30 days  |
| `Difficulty_Walking`      | 1 = Difficulty walking/climbing stairs, 0 = No              |
| `Sex`                     | 0 = Female, 1 = Male                                        |
| `Age_Group`               | Age categories from 1 = 18–24 to 13 = 80+                   |
| `Education_Level`         | Education levels from 1 = No school to 6 = College graduate |
| `Income`                  | Income groups from 1 = < \$10k to 8 = ≥ \$75k               |



🚀 RoadMap- What has been done in this project

- Data Collection & Cleaning:

BRFSS 2015 health dataset — cleaned nulls, handled outliers, encoded categorical features

- EDA & Statistical Testing:
Performed univariate and bivariate analysis, Chi-Square and t-tests confirmed significance of features like BMI, physical/mental health

- Feature Engineering:

One-hot encoding for categorical variables and ordinal encoding for ranked categorical variables

- Advanced EDA:

Some more feature engineering was done and advanced insights were drawn

- Baseline Models:

Logistic Regression, Decision Tree, Random Forest, KNN & XGBoost baseline models were developed. Class imbalances were appropriately handled.

- Advanced Models:

- 2 top models (Logistic Regression & XGBoost) were hypertuned, compared and evaluated based on metrics like Classification report (F1 score, precision & recall), confusion matrix, ROC-AUC Curve & PR-Curve

📌 

Methodology for Modeling:

- Addressed class imbalance issue.
- Normalized the dataset
- Use Stratified train-split 
- Explored LR, DT, RF, KNN & XGBoost with upsampling technique as well as class weighted approach
- Evaluated the model using accuracy, F1-score, cross-validation, confusion matrix, and AUC-ROC


Technology App:

- Use Streamlit App was developed for best performing model. This app takes input features from users and then predicts if someone is diabetic or not based.

📎Appendix

* `data` 

- Contains a link to the **original BRFSS 2015 dataset** (hosted on [CDC](https://www.cdc.gov/brfss/index.html))
- Includes saved versions of:
    - `diabetes_cleaned.csv`: cleaned dataset with raw values
    - `diabetes_encoded.csv`: final dataset used for modeling
 - Data kept below 10MB for reproducibility

* `notebooks`
    - EDA-Diabetes CDC

* `docs`
    - Sprint 1- Presentation
    - Sprint 2- Presentation
    - Sprint 3- Presentation
	

* `references`

    - https://www.cdc.gov/diabetes/php/data-research/index.html
    - https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
    - https://diabetes.org/newsroom/press-releases/new-american-diabetes-association-report-finds-annual-costs-diabetes-be	


* `conda.yml`
    - Conda environment specification

* `README.md`
    - Project landing page (Diabetes_README)

* `LICENSE` 
    - Project license


