import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

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

# parameter optimization
learning_rates = [1e-4, 1e-5, 1e-6]
iteration_nums = [1000, 2000, 3000]

models = []
train_MSEs = []
loss_hists = []
it_nums = []

# trials, logs, and plots
with open("log_trials_part2.txt", "w") as log:
    log.write("Learning_Rate, Iterations, Train_MSE, Test_MSE\n")

    for lr in learning_rates:
        for iterations in iteration_nums:
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
            for iteration in range(iterations):
                model.partial_fit(X_train, y_train_scaled)
                
                if (iteration + 1) % 50 == 0:
                    train_pred = model.predict(X_train) * y_std + y_mean
                    loss_history.append(mean_squared_error(y_train, train_pred))

            final_train_mse = loss_history[-1]
            test_pred_trial = model.predict(X_test) * y_std + y_mean
            final_test_mse = mean_squared_error(y_test, test_pred_trial)
            
            models.append(model)
            train_MSEs.append(final_train_mse)
            loss_hists.append(loss_history)
            it_nums.append(iterations)

            log.write(f"{lr}, {iterations}, {final_train_mse:.2f}, {final_test_mse:.2f}\n")

    best_idx = np.argmin(train_MSEs)
    optimal_model = models[best_idx]
    
    test_predictions = optimal_model.predict(X_test) * y_std + y_mean
    final_mse = mean_squared_error(y_test, test_predictions)
    final_r2 = r2_score(y_test, test_predictions)
    final_evs = explained_variance_score(y_test, test_predictions)

    log.write(f"\nBest Parameters Report:\n")
    log.write(f"LR: {optimal_model.eta0}, Iterations: {it_nums[best_idx]}, Test MSE: {final_mse:.2f}, R2: {final_r2:.4f}\n")

print(f"Optimal LR: {optimal_model.eta0} ; Best Test MSE: {final_mse:.2f}")
print(f"R2 Score: {final_r2:.4f} ; Explained Variance: {final_evs:.4f}")
print("\nOptimal Weight Coefficients:")
print(optimal_model.coef_)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(50, it_nums[best_idx] + 1, 50), loss_hists[best_idx], marker='o')
plt.title(f"Optimal Model Convergence (LR={optimal_model.eta0})")
plt.xlabel("Iterations")
plt.ylabel("Training MSE")
plt.grid(True)
plt.show()