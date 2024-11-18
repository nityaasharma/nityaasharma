# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('amazon_products.csv')

# Data Cleaning: Handle missing values
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Price'].fillna(df['Price'].median(), inplace=True)
df['Rating'].fillna(df['Rating'].median(), inplace=True)
df.dropna(subset=['Rating'], inplace=True)

# Visualizations: Box plot of ratings, histogram of price, and scatter plot of price vs. rating
sns.boxplot(x='Rating', data=df)
plt.title('Box Plot of Ratings')
plt.show()

sns.histplot(df['Price'], bins=30, kde=True, color='blue')
plt.title('Histogram of Product Prices')
plt.xlabel('Price')
plt.show()

sns.scatterplot(x='Price', y='Rating', data=df)
plt.title('Price vs Rating')
plt.show()

# Prepare features and target variable
X = df[['Price', 'Number of Reviews']]  # Independent features
y = df['Rating']  # Dependent feature

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics and coefficients
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
print(f'Intercept: {model.intercept_}')
print(f'Coefficients: {model.coef_}')
