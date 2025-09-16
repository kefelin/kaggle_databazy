#Alex Novák 
#Projekt: Uber Analytics

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#1nacitat data
data = pd.read_csv(r"C:\Users\Admin\Desktop\September 2025\Databaza\Uber\ncr_ride_bookings.csv")

#2. Audit
print("Tvar: ", data.shape)
print(data.info())
print(data.isnull().sum().sort_values(ascending=False).head(20))
print(data.head())

#3.vypln tie chybauce hodnoty
num_cols = data.select_dtypes(include="number").columns
cat_cols = data.select_dtypes(include="object").columns

data[num_cols] = data[num_cols].fillna(data[num_cols].median(numeric_only=True))
for c in cat_cols:
    data[c] = data[c].fillna("Unknown")

#4 Konverzia datumov
if 'Trip Date' in data.columns:
    data['Trip Date'] = pd.to_datetime(data['Trip Date'], errors='coerce')

#5 odstranit outliery pred metrikami
sns.boxplot(x=data["Booking Value"])
plt.show()

Q1 = data["Booking Value"].quantile(0.25)
Q3 = data["Booking Value"].quantile(0.75)
IQR = Q3 - Q1
before = len(data)
data = data[
    (data["Booking Value"] >= Q1 - 1.5 * IQR) &
    (data["Booking Value"] <= Q3 + 1.5 * IQR)
]
after = len(data)
print(f"Odstránených riadkov kvôli outlierom: {before - after}")

#6 zakladene metriky o stave obejdanvky
total_bookings = data.shape[0]
status_counts = data['Booking Status'].value_counts()
completed = (data['Booking Status'] == 'Completed').sum()
completion_rate = round(100 * completed / total_bookings, 2)
cancel_cust = data.get('Cancelled Rides by Customer', pd.Series([0]*len(data))).sum()
cancel_driver = data.get('Cancelled Rides by Driver', pd.Series([0]*len(data))).sum()
incomplete = data.get('Incomplete Rides', pd.Series([0]*len(data))).sum()

print('Total bookings:', total_bookings)
print('Status counts:\n', status_counts)
print('Completion rate (%):', completion_rate)
print('Cancelled by customer:', cancel_cust)
print('Cancelled by driver:', cancel_driver)
print('Incomplete rides:', incomplete)

#7 trend trzieb
if 'Trip Date' in data.columns:
    daily = data.groupby(pd.Grouper(key='Trip Date', freq='D')).agg(
        bookings=('Booking ID', 'count'),
        revenue=('Booking Value', 'sum'),
        completed=('Booking Status', lambda s: (s == 'Completed').sum())
    )
    daily['conversion'] = (daily['completed'] / daily['bookings']).fillna(0)
    daily[['bookings', 'revenue', 'conversion']].plot(
        figsize=(12, 5), subplots=True,
        title=['Bookings', 'Revenue', 'Conversion']
    )
    plt.show()

#8 heatmapa podla hodiny a dna
if 'Trip Date' in data.columns:
    data['hour'] = data['Trip Date'].dt.hour
    data['dow'] = data['Trip Date'].dt.day_name()
    pivot = data.pivot_table(index='dow', columns='hour', values='Booking ID', aggfunc='count').fillna(0)
    sns.heatmap(pivot, annot=False)
    plt.show()

# 9 dovody zrusneniaaa
if 'Reason for cancelling by Customer' in data.columns:
    print(data['Reason for cancelling by Customer'].value_counts().head(10))
if 'Driver Cancellation Reason' in data.columns:
    print(data['Driver Cancellation Reason'].value_counts().head(10))

#10 ulozenie vystupov
data.to_csv("cleaned_uber_data.csv", index=False)
plt.savefig('last_figure.png')  # uloží posledný graf

# --- Ak chceš pokračovať modelovaním ---
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# zakodovanie kategorii
X = pd.get_dummies(data.drop('Booking Value', axis=1), drop_first=True)
y = data['Booking Value']

# škálovanie číselných stĺpcov
num_cols = X.select_dtypes(include='number').columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
