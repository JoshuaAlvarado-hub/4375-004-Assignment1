import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

data_url = "https://raw.githubusercontent.com/JoshuaAlvarado-hub/4375-004-Assignment1/main/apartments.csv"
df = pd.read_csv(data_url, sep=None, engine='python', on_bad_lines='skip')
df.columns = df.columns.str.strip()

# preprocess data
df = df.dropna(subset=['price', 'bathrooms', 'bedrooms', 'square_feet'])

# drop insignificant rows
for col in ['price', 'bathrooms', 'bedrooms', 'square_feet']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# remove redundant rows
df = df.dropna(subset=['price', 'bathrooms', 'bedrooms', 'square_feet'])
df = df.drop_duplicates()

# convert categorical variables to numerical
df['has_photo'] = df['has_photo'].apply(lambda x: 1 if x != 'No' else 0)
df = pd.get_dummies(df, columns=['state', 'pets_allowed'], drop_first=True)

# columns to exclude
cols_to_drop = ['price', 'id', 'category', 'title', 'body', 'amenities',
                'currency', 'fee', 'price_display', 'price_type',
                'address', 'cityname', 'source', 'time', 'latitude', 'longitude']

existing_drops = [c for c in cols_to_drop if c in df.columns]

X = df.drop(columns=existing_drops).select_dtypes(include=[np.number, bool]).astype(float).values
y = df['price'].values

# standardize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# trials, logs, and plots
learning_rates = [1e-4, 1e-5, 1e-6]

plt.figure(figsize=(10, 6))

with open("log_trials.txt", "w") as log:
    log.write("Learning_Rate, Epochs, Train_MSE, Test_MSE\n")

    for lr in learning_rates:
        model = SGDRegressor(
            loss="squared_error",
            max_iter=1,
            learning_rate="constant",
            eta0=lr,
            warm_start=True,
            penalty=None,
            shuffle=False
        )

        epochs = 1000
        loss_history = []

        for epoch in range(epochs):
            model.partial_fit(X_train, y_train)

            train_predictions = model.predict(X_train)
            train_mse = mean_squared_error(y_train, train_predictions)
            loss_history.append(train_mse)

        test_predictions = model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_predictions)

        log.write(f"{lr}, 1000, {train_mse:.2f}, {test_mse:.2f}\n")
        plt.plot(range(1000), loss_history, label=f"LR: {lr}")

plt.title("MSE vs. Number of Epochs (SGDRegressor)")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
