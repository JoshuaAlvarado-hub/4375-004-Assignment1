import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# retrieve data (apartment prices)
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
iteration_nums = [1000, 2000, 3000]

with open("log_trials_part1.txt", "w") as log:
    # training models
    log.write("TRAINING\n")
    log.write("Learning_Rate, Iterations, Train_MSE\n")

    # For models generated during training:
    models = []  # will store models
    train_MSEs = []  # will store MSEs
    loss_hists = []  # will store loss histories
    it_nums = []  # will store numbers of iterations
    
    # training models
    for lr in learning_rates:
        for iterations in iteration_nums:
            model = LinearRegressionGD(learning_rate=lr, iterations=iterations)
            model.fit(X_train, y_train)

            # tracking models' information
            models.append(model)
            train_MSEs.append(model.loss_history[-1])
            loss_hists.append(model.loss_history)
            it_nums.append(iterations)

            log.write(f"{lr}, {iterations}, {train_MSEs[-1]:.2f}\n")

    # determining optimal model
    min_train_mse = min(train_MSEs)
    min_train_mse_index = train_MSEs.index(min_train_mse)

    # storing information of optimal model
    optimal_model = models[min_train_mse_index]  # model
    optimal_model_lr = optimal_model.lr  # learning rate
    optimal_model_iterations = it_nums[min_train_mse_index]  # number of iterations
    optimal_model_loss_history = loss_hists[min_train_mse_index]

    # testing optimal model
    log.write(f"\nTESTING OPTIMAL MODEL\n")

    predictions = optimal_model.predict(X_test)
    test_mse = np.mean((predictions - y_test) ** 2)

    # track testing MSE of optimal model
    log.write("Learning_Rate, Iterations, Test_MSE\n")
    log.write(f"{optimal_model_lr}, {optimal_model_iterations}, {test_mse:.2f}\n")

    # x-axis: recorded iterations
    iterations_recorded = np.arange(1, optimal_model_iterations + 1)

    # graph of training history for optimal model
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations_recorded, optimal_model_loss_history, marker='o')
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Training MSE")
    ax.set_title(
        f"Training MSE vs. Iterations for Optimal Model\n(LR={optimal_model_lr}, Iterations={optimal_model_iterations})")
    ax.grid(True)
