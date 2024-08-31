import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

white_wine = pd.read_csv('winequality-red.csv')

print(white_wine.head())

print(white_wine.isnull().sum())

white_wine.dropna(inplace=True)

print(white_wine.describe())

sns.countplot(x='quality', data=white_wine)
plt.title('White Wine Quality Distribution')
plt.show()

sns.pairplot(white_wine, hue='quality')
plt.show()

X_white = white_wine.drop('quality', axis=1)
y_white = white_wine['quality']

X_train, X_test, y_train, y_test = train_test_split(X_white, y_white, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

importance = model.feature_importances_
features = X_white.columns

plt.barh(features, importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Predicting White Wine Quality')
plt.show()
