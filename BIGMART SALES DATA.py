import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')

print("Train Dataset:\n", train_df.head())
print("\nTest Dataset:\n", test_df.head())

train_df['Item_Weight'].fillna(train_df['Item_Weight'].mean(), inplace=True)
test_df['Item_Weight'].fillna(test_df['Item_Weight'].mean(), inplace=True)

train_df['Outlet_Size'].fillna(train_df['Outlet_Size'].mode()[0], inplace=True)
test_df['Outlet_Size'].fillna(test_df['Outlet_Size'].mode()[0], inplace=True)

train_df = pd.get_dummies(train_df, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)

test_df = test_df.reindex(columns = train_df.columns, fill_value=0)
test_df = test_df.drop('Item_Outlet_Sales', axis=1, errors='ignore')

X = train_df.drop('Item_Outlet_Sales', axis=1)  # Target column
y = train_df['Item_Outlet_Sales']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_df_scaled = scaler.transform(test_df)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

y_val_pred_lr = lr.predict(X_val_scaled)
print("Linear Regression Model")
print("MAE:", mean_absolute_error(y_val, y_val_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_val_pred_lr)))
print("R2 Score:", r2_score(y_val, y_val_pred_lr))

y_val_pred_rf = rf.predict(X_val_scaled)
print("\nRandom Forest Model")
print("MAE:", mean_absolute_error(y_val, y_val_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_val_pred_rf)))
print("R2 Score:", r2_score(y_val, y_val_pred_rf))

y_test_pred_rf = rf.predict(test_df_scaled)
y_test_pred_lr = lr.predict(test_df_scaled)

output_rf = pd.DataFrame({'Id': test_df.index, 'Predicted_RF': y_test_pred_rf})
output_lr = pd.DataFrame({'Id': test_df.index, 'Predicted_LR': y_test_pred_lr})
output_rf.to_csv('predictions_rf.csv', index=False)
output_lr.to_csv('predictions_lr.csv', index=False)
print("Predictions saved to 'predictions_rf.csv' and 'predictions_lr.csv'")

plt.figure(figsize=(10, 6))
plt.scatter(np.log1p(y_val), np.log1p(y_val_pred_lr), color='green', label='Predicted', alpha=0.6)
plt.plot([np.log1p(min(y_val)), np.log1p(max(y_val))], 
         [np.log1p(min(y_val)), np.log1p(max(y_val))], 
         color='red', lw=2, label='Actual')
plt.title('Linear Regression: Actual vs Predicted (Log-Scaled)')
plt.xlabel('Log-Scaled Actual')
plt.ylabel('Log-Scaled Predicted')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(np.log1p(y_val), np.log1p(y_val_pred_rf), color='blue', label='Predicted', alpha=0.6)
plt.plot([np.log1p(min(y_val)), np.log1p(max(y_val))], 
         [np.log1p(min(y_val)), np.log1p(max(y_val))], 
         color='red', lw=2, label='Actual')
plt.title('Random Forest: Actual vs Predicted (Log-Scaled)')
plt.xlabel('Log-Scaled Actual')
plt.ylabel('Log-Scaled Predicted')
plt.legend()
plt.show()
