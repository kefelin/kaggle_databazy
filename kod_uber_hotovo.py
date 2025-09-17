# Alex Novák 
# Projekt: Uber Analytics

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Načítať data
data = pd.read_csv("E:\\z netu\\databaza\\ncr_ride_bookings.csv")

# 2. Audit
print("Tvar: ", data.shape)
print(data.info())
print(data.isnull().sum().sort_values(ascending=False).head(20))
print(data.head())

# 3. Vyplniť chýbajúce hodnoty
num_cols = data.select_dtypes(include="number").columns
cat_cols = data.select_dtypes(include="object").columns

data[num_cols] = data[num_cols].fillna(data[num_cols].median(numeric_only=True))
for c in cat_cols:
    data[c] = data[c].fillna("Unknown")

# 4. Konverzia dátumov
if 'Trip Date' in data.columns:
    data['Trip Date'] = pd.to_datetime(data['Trip Date'], errors='coerce')

# 5. Odstrániť outliery pred metrikami
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

# 6. Základné metriky o stave objednávky
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

# 7. Trend tržieb
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

# 8. Heatmapa podľa hodiny a dňa
if 'Trip Date' in data.columns:
    data['hour'] = data['Trip Date'].dt.hour
    data['dow'] = data['Trip Date'].dt.day_name()
    pivot = data.pivot_table(index='dow', columns='hour', values='Booking ID', aggfunc='count').fillna(0)
    sns.heatmap(pivot, annot=False)
    plt.show()

# 9. Dôvody zrušenia
if 'Reason for cancelling by Customer' in data.columns:
    print(data['Reason for cancelling by Customer'].value_counts().head(10))
if 'Driver Cancellation Reason' in data.columns:
    print(data['Driver Cancellation Reason'].value_counts().head(10))

#Konverzný lievik podľa mesta/okresu
if 'City' in data.columns:
    funnel = data.groupby('City').agg(
        booked=('Booking ID', 'count'),
        completed=('Booking Status', lambda s: (s == 'Completed').sum())
    )
    funnel['conversion_rate'] = round(100 * funnel['completed'] / funnel['booked'], 2)
    funnel = funnel.sort_values('conversion_rate', ascending=False)
    print("\nKonverzný lievik podľa mesta/okresu:")
    print(funnel.head(10))

    plt.figure(figsize=(10, 5))
    sns.barplot(x=funnel.index, y=funnel['conversion_rate'], palette='viridis')
    plt.xticks(rotation=45)
    plt.title('Conversion Rate (%) by City')
    plt.ylabel('Conversion Rate (%)')
    plt.xlabel('City')
    plt.tight_layout()
    plt.savefig('conversion_by_city.png')
    plt.show()
else:
    print("\nStĺpec 'City' sa v dátach nenašiel – úloha 1 preskočená.")

#Porovnanie metrík podľa spôsobu platby
if 'Payment Method' in data.columns:
    cancel_by_payment = data.groupby('Payment Method').agg(
        total_bookings=('Booking ID', 'count'),
        cancelled=('Booking Status', lambda s: (s != 'Completed').sum())
    )
    cancel_by_payment['cancel_rate'] = round(100 * cancel_by_payment['cancelled'] / cancel_by_payment['total_bookings'], 2)
    cancel_by_payment = cancel_by_payment.sort_values('cancel_rate', ascending=False)
    print("\nMiera zrušení podľa spôsobu platby:")
    print(cancel_by_payment)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=cancel_by_payment.index, y=cancel_by_payment['cancel_rate'], palette='magma')
    plt.title('Cancellation Rate (%) by Payment Method')
    plt.ylabel('Cancellation Rate (%)')
    plt.xlabel('Payment Method')
    plt.tight_layout()
    plt.savefig('cancel_rate_by_payment.png')
    plt.show()
else:
    print("\nStĺpec 'Payment Method' sa v dátach nenašiel – úloha 2 preskočená.")

# 10. Uloženie výstupov
data.to_csv("cleaned_uber_data.csv", index=False)
plt.savefig('last_figure.png')  # uloží posledný graf

# Modelovanie
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

high_card_cols = [col for col in data.select_dtypes(include='object').columns if data[col].nunique() > 100]
data_model = data.drop(columns=high_card_cols + ['Booking Value'])

X = pd.get_dummies(data_model, drop_first=True)
y = data['Booking Value']

num_cols = X.select_dtypes(include='number').columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Rozdelenie
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
