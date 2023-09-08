# Titanic_psg_Survivability
Certainly! The provided code performs data analysis and builds a machine learning model to predict survival on the Titanic dataset. Here's a description of each part of the code:

1. **Data Loading:**
   - The code starts by importing necessary libraries, including NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn.
   - It loads the Titanic dataset from a CSV file ('train.csv') into a Pandas DataFrame called `titanic_data`.

2. **Data Exploration and Preprocessing:**
   - It displays the first 5 rows of the dataset using `titanic_data.head()` to provide a glimpse of the data.
   - The code checks the shape of the dataset using `titanic_data.shape` to see the number of rows and columns.
   - `titanic_data.info()` is used to get information about the data types and non-null counts in each column.
   - It checks for missing values in the dataset using `titanic_data.isnull().sum()`.

3. **Data Cleaning:**
   - The 'Cabin' column, which contains many missing values, is dropped from the DataFrame.
   - Missing values in the 'Age' column are replaced with the mean age of passengers.
   - The mode value of the 'Embarked' column is found and used to fill missing values in that column.
   - Another check for missing values is performed to ensure all missing data is handled.

4. **Data Analysis:**
   - The code provides some basic statistics about the dataset using `titanic_data.describe()`.
   - It counts the number of passengers who survived and those who did not using `titanic_data['Survived'].value_counts()`.

5. **Data Visualization:**
   - Seaborn is used for data visualization.
   - A count plot of the 'Survived' column is created using `sns.countplot(x='Survived', data=titanic_data)`.
   - Count plots are generated for gender-wise survival, passenger class distribution, and class-wise survival.
   - Categorical values like 'Sex' and 'Embarked' are converted to numerical values for modeling purposes.

6. **Feature Selection and Target Label Separation:**
   - Features are separated from the target label ('Survived') to prepare the data for modeling.
   - The features are stored in the `X` DataFrame, and the target label is stored in the `Y` Series.

7. **Data Splitting:**
   - The dataset is split into training and testing sets using `train_test_split` from scikit-learn. The split is 80% training and 20% testing.

8. **Model Training:**
   - A Logistic Regression model is created with `max_iter=10000` (a high number of iterations to ensure convergence).
   - The model is trained on the training data using `model.fit(X_train, Y_train)`.

9. **Model Evaluation:**
   - The code predicts survival on both the training and test datasets.
   - Accuracy scores are calculated for both the training and test data using `accuracy_score`.
   - The accuracy scores are printed to evaluate the performance of the Logistic Regression model.

Overall, this code performs data preprocessing, visualization, and builds a Logistic Regression model to predict passenger survival on the Titanic dataset. The accuracy scores on the training and test data are used to assess the model's performance.
