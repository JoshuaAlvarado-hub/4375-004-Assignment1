import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.n_iter = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iter):
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            mse = np.mean((y_pred - y)**2)
            self.loss_history.append(mse)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
# trials, logs, and plots
learning_rates = [0.1, 0.01, 0.001]

plt.figure(figsize=(10, 6))

with open("log_trials.txt", "w") as log:
    log.write("Learning_Rate, Iterations, Train_MSE, Test_MSE\n")
    
    for lr in learning_rates:
        model = LinearRegressionGD(learning_rate=lr, iterations=1000)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        test_mse = np.mean((predictions - y_test)**2)
        train_mse = model.loss_history[-1]
        
        log.write(f"{lr}, 1000, {train_mse:.2f}, {test_mse:.2f}\n")
        plt.plot(range(1000), model.loss_history, label=f"LR: {lr}")

plt.title("MSE vs. Number of Iterations (Manual GD)")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()