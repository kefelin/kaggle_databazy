
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Load data
data = pd.read_csv("/kaggle/input/uber-ride-analytics-dashboard/ncr_ride_bookings.csv")
print(data)

# Basic data info
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Fill missing numerical values with median
data[data.select_dtypes(include='number').columns] = data.select_dtypes(include='number').fillna(data.median(numeric_only=True))

# Check missing values again
print(data.isnull().sum())

# Remove outliers from Booking Value
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=data["Booking Value"])
plt.show()

Q1 = data["Booking Value"].quantile(0.25)
Q3 = data["Booking Value"].quantile(0.75)
IQR = Q3 - Q1
data = data[
    (data["Booking Value"] >= Q1 - 1.5 * IQR) &
    (data["Booking Value"] <= Q3 + 1.5 * IQR)
]

# Show categorical columns
categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
print("Categorical columns:", categorical_cols)

# Prepare data for machine learning
from sklearn.model_selection import train_test_split

X = data.drop('Booking Value', axis=1)
y = data['Booking Value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create summary statistics tables
print("\n=== SUMMARY STATISTICS ===")

# Total bookings
total_bookings = data.shape[0]
print(f"Total Bookings: {total_bookings}")

# Booking status counts
booking_status = data['Booking Status'].value_counts()
print("\nBooking Status Distribution:")
print(booking_status)

# Completed rides
completed_rides = data[data['Booking Status'] == 'Completed'].shape[0]
completion_rate = (completed_rides / total_bookings) * 100
print(f"\nCompleted Rides: {completed_rides}")
print(f"Completion Rate: {completion_rate:.2f}%")

# Cancelled by customer
cancelled_by_customer = data['Cancelled Rides by Customer'].sum()
print(f"\nCancelled by Customer: {cancelled_by_customer}")

# Cancelled by driver
cancelled_by_driver = data['Cancelled Rides by Driver'].sum()
print(f"Cancelled by Driver: {cancelled_by_driver}")

# Incomplete rides
incomplete_rides = data['Incomplete Rides'].sum()
print(f"Incomplete Rides: {incomplete_rides}")

# Reasons for cancellation
if 'Reason for cancelling by Customer' in data.columns:
    customer_cancel_reasons = data['Reason for cancelling by Customer'].value_counts()
    print("\nCustomer Cancellation Reasons:")
    print(customer_cancel_reasons)

if 'Driver Cancellation Reason' in data.columns:
    driver_cancel_reasons = data['Driver Cancellation Reason'].value_counts()
    print("\nDriver Cancellation Reasons:")
    print(driver_cancel_reasons)