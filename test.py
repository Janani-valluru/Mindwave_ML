import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import os
from sklearn.model_selection import cross_val_score
import pickle
import joblib


# Reading the dataset
df = pd.read_csv("C:/Users/JANANI.V.A/Desktop/mentalhealth/ML/new1.csv")

# Calculate average score per user
user_average_scores = df.groupby('username')['score'].mean().reset_index()
print(user_average_scores)

# Defining the depression levels based on average score
def get_depression_level_average(average_score):
    if average_score >= 0 and average_score <= 6:
        return "low level"
    elif average_score >= 7 and average_score <= 12:
        return "Moderate level"
    else:
        return "High level"

# Apply the function to categorize users based on average score
user_average_scores['depression_level'] = user_average_scores['score'].apply(get_depression_level_average)

# Encoding categorical variables (if needed for further analysis)
encoder = LabelEncoder()
user_average_scores['username_encoded'] = encoder.fit_transform(user_average_scores['username'])

pickle.dump(encoder, open('username_encoded.pickle', 'wb'))

print(user_average_scores)

## PRE-PROCESSING (If needed for further analysis or model training)

# Checking for missing values
missing_values = user_average_scores.isnull().sum()
print("Missing values:/n", missing_values)

# Impute missing values if any
imputer = SimpleImputer(strategy='most_frequent')
data_imputed = pd.DataFrame(imputer.fit_transform(user_average_scores), columns=user_average_scores.columns)

# Droping original categorical columns
data_encoded = data_imputed

print(data_encoded.head())
print(data_encoded.size)

X = data_encoded[['username_encoded', 'score']]  # Features
y = data_encoded['depression_level']  # Target variable

# Store the data_encoded DataFrame with selected columns into a new JSON file
data_encoded_selected = data_encoded[['username_encoded', 'score']]
json_filename = 'encoded_data.json'
data_encoded_selected.to_json(json_filename, orient='records')

print(f"Encoded data saved to {json_filename}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# RandomForestClassifier with hyperparameter tuning
rf_classifier = RandomForestClassifier()

param_grid = {
    'n_estimators': [50,100, 200, 300],  
    'max_depth': [None, 5, 20],       
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4]     
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the random forest classifier with the best hyperparameters
best_rf_classifier = RandomForestClassifier(**best_params)
best_rf_classifier.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = best_rf_classifier.predict(X_test)


bias_variance_scores = cross_val_score(best_rf_classifier, X_train, y_train, cv=10, scoring='accuracy')

print("Best parmas : ")
print(best_params)



# best_rf_classifier is the final model
filename = 'depression_prediction_model_v1.joblib'
#pickle.dump(best_rf_classifier, open(filename, 'wb'))
joblib.dump(best_rf_classifier, filename)

# Calculate average bias and variance
average_bias = 1 - np.mean(bias_variance_scores)
average_variance = np.var(bias_variance_scores)

print(f"/nBias-Variance Analysis:")
print(f"Average Bias: {average_bias:.4f}")
print(f"Average Variance: {average_variance:.4f}")

# Interpretation
if average_bias > 0.1 and average_variance > 0.1:
    print("Model likely suffers from both high bias and high variance.")
elif average_bias > 0.1:
    print("Model likely suffers from high bias (underfitting).")
elif average_variance > 0.1:
    print("Model likely suffers from high variance (overfitting).")
else:
    print("Model seems to have low bias and low variance.")

# Evaluate the model
print("/nRandom Forest Classifier with Hyperparameter Tuning:/n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Additional analysis (e.g., feature importance)
print("Feature Importance:")
feature_importance = best_rf_classifier.feature_importances_
print(feature_importance)



# Define the path to the result folder
result_folder = "demo/result"

# Check if the result folder exists, if not, create it
if not os.path.exists(result_folder):
    os.makedirs(result_folder)



for category in data_encoded['depression_level'].unique():
    # Filter the dataset for the current category
    filtered_data = data_encoded[data_encoded['depression_level'] == category]
    # Define the filename for the CSV file based on the category and the result folder
    filename = os.path.join(result_folder, f"test_{category.replace(' ', '_').lower()}_data.csv")
    # Save the filtered dataset to a CSV file
    filtered_data.to_csv(filename, index=False)
    print(f"Saved {filename} successfully.")


# Combining the csv files


# Get a list of CSV files in the result folder
csv_files = [f for f in os.listdir(result_folder) if f.endswith('.csv')]

# Create an empty list to store the DataFrames
dataframes = []

# Iterate over the CSV files and read them into DataFrames
for file in csv_files:
    filepath = os.path.join(result_folder, file)
    df = pd.read_csv(filepath)
    dataframes.append(df)

# Concatenate the DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_filename = os.path.join(result_folder, "combined_data.csv")
combined_df.to_csv(combined_filename, index=False)

print(f"Combined data saved to {combined_filename}")


# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_rf_classifier.classes_, yticklabels=best_rf_classifier.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()


train_sizes, train_scores, test_scores = learning_curve(best_rf_classifier, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.show()


