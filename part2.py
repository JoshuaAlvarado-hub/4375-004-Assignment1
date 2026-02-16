import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

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

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize X
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# scale y
y_mean = y_train.mean()
y_std = y_train.std()
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# parameters to be varied
learning_rates = [1e-5, 1e-6, 1e-7]
iteration_nums = [1000, 2000, 3000]  # must be multiples of 50

plt.figure(figsize=(10, 6))

# trials, logs, and plots
with open("log_trials_part2.txt", "w") as log:
    # training models
    log.write("TRAINING\n")
    log.write("Learning_Rate, Iterations, Train_MSE\n")

    # For models generated during training:
    models = []      # will store models
    train_MSEs = []  # will store MSEs
    loss_hists = []  # will store loss histories
    it_nums = []     # will store numbers of iterations

    # varying learning rates and numbers of iterations for each model
    for lr in learning_rates:
        for iterations in iteration_nums:
            # Create SGDRegressor model using parameters
            model = SGDRegressor(
                loss="squared_error",
                max_iter=1,
                learning_rate="constant",
                eta0=lr,
                warm_start=True,
                penalty=None,
                shuffle=False
            )

            loss_history = []

            # training the model
            for iteration in range(iterations):
                # partial fit each iteration
                model.partial_fit(X_train, y_train_scaled)

                # track training MSE for every 50th iteration
                if (iteration + 1) % 50 == 0:
                    train_predictions = model.predict(X_train)
                    train_predictions_orig = train_predictions * y_std + y_mean
                    train_mse = mean_squared_error(y_train, train_predictions_orig)
                    loss_history.append(train_mse)

            # store relevant information after training:
            models.append(model)                 # model itself
            train_MSEs.append(loss_history[-1])  # training MSE
            loss_hists.append(loss_history)      # loss history
            it_nums.append(iterations)           # number of iterations

            # track in log file
            log.write(f"{lr}, {iterations}, {train_MSEs[-1]:.2f}\n")

    # determining optimal model
    min_train_mse = min(train_MSEs)
    min_train_mse_index = train_MSEs.index(min_train_mse)

    # storing information of optimal model
    optimal_model = models[min_train_mse_index]                   # model
    optimal_model_lr = optimal_model.eta0                         # learning rate
    optimal_model_iterations = it_nums[min_train_mse_index]       # number of iterations
    optimal_model_loss_history = loss_hists[min_train_mse_index]  # loss history

    # testing optimal model
    log.write(f"\nTESTING OPTIMAL MODEL\n")

    test_predictions = optimal_model.predict(X_test) * y_std + y_mean
    test_mse = mean_squared_error(y_test, test_predictions)

    # track testing MSE of optimal model
    log.write("Learning_Rate, Iterations, Test_MSE\n")
    log.write(f"{optimal_model_lr}, {optimal_model_iterations}, {test_mse:.2f}\n")
