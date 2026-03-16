# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Collect the dataset containing nutritional features of food items such as sugar, carbohydrates, calories, and fiber.
2. Preprocess the data by cleaning, normalizing the features, and splitting the dataset into training and testing sets.
3. Train a **Support Vector Machine (SVM)** model using the training dataset to classify food items based on their nutritional values.
4. Optimize the model by tuning hyperparameters such as kernel type, C value, and gamma using techniques like Grid Search or Cross Validation.
5. Evaluate the optimized model on the test dataset and measure accuracy to determine the classification performance.


## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features = ['Calories', 'Total Fat', 'Saturated Fat', 'Sugars', 'Dietary Fiber', 'Protein']
target = 'class'

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm = SVC()

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Name: PARUVATHA VARSHINI PS")
print("Register Number: 212225100033")
print("Accuracy:", accuracy)

print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

```

## Output:
<img width="831" height="771" alt="image" src="https://github.com/user-attachments/assets/544917da-9dfd-4d52-afc6-6505b56b5bfd" />
<img width="896" height="777" alt="image" src="https://github.com/user-attachments/assets/65c7a256-8bf4-4aa5-94cd-b18fc8ad0124" />

## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
