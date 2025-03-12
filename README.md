# Segment-based-Factor-Analysis-with-ML-Regression-And-XAI-Implementation

This project analyzes a dataset containing hospital inpatient discharge records. The primary objective is to clean and preprocess the data for further analysis, including handling missing values, converting variables, removing outliers, and conducting statistical tests. After which, a brief EDA process is focused on the dependent variable, Length of Stay (LoS) with relationships against the other independent variables explored, which in turn gives a small glimpse into the significant factors affecting LoS for both subgroups before the modelling phase.

## Project Overview

The dataset contains information about hospital admissions, including details such as length of stay, diagnosis codes, procedures, and charges. This project focuses on data cleaning, normalization, statistical analysis, data visualizations, statistical modelling and explainable AI with the final goal centered around differences between importances of significant factors affecting LoS for elective and emergency admissions.

## Libraries Used

- `pandas`: For data manipulation and cleaning
- `matplotlib`: For data visualizations and supplementing them with further applications.
- `scipy.stats`: For performing statistical tests (e.g., T-test)
- `sklearn`: For several applications, including machine learning models, cross validation, etc.
- `seaborn`: For data visualizations.
- `shap`: For explainability analysis
- `tensorflow`: For implementing Neural Networks
- `xgboost`: For implementing XGBoost
- `statsmodels`: For the purpose of implementing Variance Inflation Factor tests 

## Steps

1. **Data Importing & Exploration**
   - The dataset is loaded using `pandas.read_csv()`.
   - The first 100 rows are displayed to get an overview of the data.
   
2. **Data Cleaning**
   - Convert specific columns (e.g., `float64` columns) to integer types.
   - Handle missing values using various strategies, including:
     - Dropping rows with missing values in critical columns.
     - Visualizing missing data.
   
3. **Handling Outliers**
   - The IQR method is used to identify outliers in the `Total Charges` column.
   - Anomaly detection is applied using the `IsolationForest` algorithm.

4. **Normalization and Conversion**
   - Normalize the `Zip Code - 3 digits` column to eliminate unnecessary decimal points and correct incorrect entries.
   - Convert `Length of Stay` from categorical to numerical format for regression analysis.

5. **Statistical Analysis**
   - A T-test is performed to compare the means of `Length of Stay` between elective and emergency admissions to determine if the differences are statistically significant.

6. **Data Splitting**
   - The dataset is split into two subgroups: **Elective** and **Emergency** admissions.
   - These subgroups are saved into separate CSV files for further analysis.

7. **Data Overview**
   - The first 10 rows and the data types of both subgroups are displayed for initial inspection.

8. **Categorical Columns Analysis**
   - The unique number of categories for each categorical column in both subgroups are computed and printed.

9. **Length of Stay (LoS) Distribution**
   - The distribution of Length of Stay (LoS) for both elective and emergency admissions is visualized using kernel density estimates.
   - Additionally, the percentages of days for LoS are computed and displayed.

10. **Correlation Analysis Using Correlation Ratio**
   - The correlation ratio (eta squared) is used to assess the relationship between categorical variables and LoS for both subgroups. The `correlation_ratio` function calculates this for each categorical column.
   - The correlation results are displayed for both elective and emergency admissions.

11. **Pearson's Correlation Matrix**
   - A correlation matrix of numerical variables is created for both datasets, with a custom color map to indicate the strength of correlations leveraging `matplotlib.colors` and `LinearSegmentedColormap`.
   - The `correlation_matrix` function generates and visualizes the correlation matrix for both elective and emergency admissions.

12. **Data Preprocessing**
   - Categorical variables are analyzed for both subgroups based on which encoding techniques work for them prior to transformation.
   - The `category_encoders` library is imported alongside `OrdinalEncoder` and `OneHotEncoder` to facilitate appropriate encoding for the variables to lead to modelling for both subgroups.
   - Multicollinearity is tested beforehand to check redundant features.

13. **Feature Selection**
   - `Recursive Feature Elimination` is employed to facilitate selection of the best half of all the features for each subgroup.
   - `Decision Trees` are set up as the base for this technique as it provides a good balance of interpretability and handling of complex relationships with the data.
   - Uncommon features ranked between both subgroups are removed for an effective comparison at the end.

14. **Predictive Modelling**
   - Several base Machine Learning models, including the likes of `Linear Regression`, `Decision Trees`, `Random Forests`, `XGBoost` and `Neural Networks` are used on both subgroups with the evaluation metrics being `R-Squared` and `RMSE` to gauge the variability of the independent variables with respect to the dependent variables.
   - Tree visualizations pertaining to Tree based models and feature importance graphs are also plotted to keep track of the most significant features.
   - The best model amongst the models tested, graded based on the evaluation metrics, being XGBoost is chosen and is applied `5-fold Cross Validation` upon to analyse the consistency of training and test scores across various folds of data.
   - The model is then carried over for more analysis for explanability.

15. **Explanability Analysis**
   - The `SHAP` library is installed and imported to assess the feature importances of XGBoost models pertaining to both subgroups.
   - Boolean columns are converted to float for `SHAP` compatibility. Afterwhich, 100 samples are selected from both the training and test data for the analysis, for both subgroups.
   - Using `TreeExplainer`, `SHAP` values are computed for the test set. It is checked whether if the `SHAP` values are for a single-output model and ensures that they are correctly formatted.
   - `SHAP` summary plots (both bar and dot plots) are created to show the feature importance and their impact on predictions.
   - Lastly, a `SHAP` waterfall plot is generated for the first test instance, illustrating the contribution of each feature to the model's prediction for both subgroups.

## Results

The feature importance plots signify the following inferences:
   - Total Charges and Total Costs, equating to patient bills and hospitalization charges, are consistently the most significant factors affecting LoS for both subgroups. This makes sense as the longer a patient stays at a hospital, so would be the increase in both expenses.
   - CCS Diagnosis Code and Operating Certificate Number, equating to gravity of illness and regional influence, are also consistently found in the top 5 most significant features across both subgroups. These also make sense as the condition of a patient and the level of treatment can all determine whether a patient stays longer at a hospital or not.
   - APR Medical Surgical Description_Surgical and APR DRG Code, equating to the presence of surgeries and disease classification, varies very signifcantly between both admission types with a higher significance in elective admissions compared to emergency admissions.
