# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
### NAME :RAGUNATH R
### REGISTER NO :212222240081
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program

### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load your dataset
df = pd.read_csv('/content/daily-minimum-temperatures-in-me.csv')  # Update the path to your dataset

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

# Convert 'Daily minimum temperatures' to numeric, forcing errors to NaN
df['Daily minimum temperatures'] = pd.to_numeric(df['Daily minimum temperatures'], errors='coerce')

# Drop rows with invalid dates or temperatures
df = df.dropna(subset=['Date', 'Daily minimum temperatures'])

# Convert 'Date' column to ordinal values
df['Date_ordinal'] = df['Date'].apply(lambda x: x.toordinal())
X = df['Date_ordinal'].values.reshape(-1, 1)
```

A - LINEAR TREND ESTIMATION
```
# Linear trend estimation
y = df['Daily minimum temperatures'].values
linear_model = LinearRegression()
linear_model.fit(X, y)
df['Linear_Trend'] = linear_model.predict(X)

# Plotting linear trend
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Daily minimum temperatures'], label='Original Data', color='blue')
plt.plot(df['Date'], df['Linear_Trend'], color='yellow', label='Linear Trend')
plt.title('Linear Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Daily Minimum Temperatures (°C)')
plt.legend()
plt.grid(True)
plt.show()
```


B- POLYNOMIAL TREND ESTIMATION
```
# Polynomial trend estimation
poly = PolynomialFeatures(degree=3)  # You can adjust the degree as needed
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
df['Polynomial_Trend'] = poly_model.predict(X_poly)

# Plotting polynomial trend
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Daily minimum temperatures'], label='Original Data', color='blue')
plt.plot(df['Date'], df['Polynomial_Trend'], color='green', label='Polynomial Trend')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Daily Minimum Temperatures (°C)')
plt.legend()
plt.grid(True)
plt.show()
```

### OUTPUT
![image](https://github.com/user-attachments/assets/179d5d1f-9aa1-4309-a587-f63c55d67137)

A - LINEAR TREND ESTIMATION
![image](https://github.com/user-attachments/assets/4b0a30dc-dee6-4c77-ac21-6a2181220671)

B- POLYNOMIAL TREND ESTIMATION
![image](https://github.com/user-attachments/assets/f5603a06-4668-4072-8cf3-0ee4d0156905)


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
