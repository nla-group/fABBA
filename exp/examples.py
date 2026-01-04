from fABBA import JABBA
import numpy as np
import matplotlib.pyplot as plt

# Simulate 20 independent univariate series
np.random.seed(0)
data = np.cumsum(np.random.randn(20, 30), axis=1)  # random walks, 20 time series, each with 800 features

jabba = JABBA(tol=0.3, init='kmeans', k=5, verbose=1)  # auto-digitization
symbols = jabba.fit_transform(data)
recon = jabba.inverse_transform(symbols)



print(jabba.parameters)
#Total number of pieces: 282
#Generate 5 symbols
#Model(centers=array([[ 1.        , -1.62393348],
#       [ 5.        , -2.09407943],
#       [ 1.31730769,  1.2972567 ],
#       [ 4.        ,  1.98348455],
#       [ 2.37681159, -1.59305148]]), alphabets=array(['A', 'a', 'B', 'b', 'C'], dtype='<U1'))

# Get embedding for the each symbol, and concatenate embeddings for the first time series
alphabets = jabba.parameters.alphabets.tolist()
centers = jabba.parameters.centers # shape: N by 2, N is dependent on compression phase and each instance is with lenth and increment
first_embedding = np.vstack([jabba.parameters.centers[alphabets.index(p)][:2] for p in symbols[0]])

# Get embeddings for the second time series
second_embedding = np.vstack([jabba.parameters.centers[alphabets.index(p)][:2] for p in symbols[1]])

# You can get together for the whole 20 time series
embeddings = list()
for i in range(data.shape[0]):
    embedding_j = np.vstack([jabba.parameters.centers[alphabets.index(p)][:2] for p in symbols[i]])
    print(embedding_j.shape)
    embeddings.append(embedding_j)

# In the next stepss, you can do the downstream time series tasks, e.g., classification or forecasting.

# Plot first 3 series
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(data[i], label='Original', alpha=0.8)
    plt.plot(recon[i], '--', label='JABBA reconstruction')
    plt.title(f'Series {i} -> compressed to {len(symbols[i])} symbols')
    plt.legend()
plt.tight_layout()
plt.show()



# For multivariate time series (N, n_channels, n_features), you can load the parameters of ABBA:

X_train = np.random.randn(100, 5, 10)
X_test  = np.random.randn(30, 5, 10)


## Setting last_dim = True
print("On train set")
jabba = JABBA(tol=0.05, last_dim = True).fit(X_train)                    # learn vocabulary
print("new shape:", jabba.new_shape)
JABBA(tol=0.05).fit_transform(X_train) 
symbols_train, starts = jabba.transform(X_train)          # use same symbols!
X_train_recon = jabba.inverse_transform(symbols_train, starts)
print("error on train set:", np.mean((X_train - X_train_recon)**2))


print(symbols_train[0]) # for the symbolic representation for the first time series - a list contains 5 lists
# [['7', '.', ',', '2', '^', '0', '5', '&'],
#  ['%', '5', '%', '-', "'", 'Q', '*', '?'],
#  ['-', 'v', '+', 'n', '2', '!'],
#  ['\\', '+', '!', '*', '.', ',', '\\'],
#  ['1', '$', '6', '"', 'f', '-', '&', '9']]

## Setting last_dim = False
print()
print("last_dim = False")
print("On train set")
jabba = JABBA(tol=0.05, last_dim = False).fit(X_train)                    # learn vocabulary
print("new shape:", jabba.new_shape)
symbols_train = JABBA(tol=0.05).fit_transform(X_train) 
symbols_train, starts = jabba.transform(X_train)          # use same symbols!
X_train_recon = jabba.inverse_transform(symbols_train, starts)
print("error on train set:", np.mean((X_train - X_train_recon)**2))


print(symbols_train[0]) # for the symbolic representation for the first time series - a list contains 5 lists
# ['\x8b', ',', '!', '1', '2', '(', '2', 'Q', '/', '8', '4', '+', '2', '+', 'o', '.', '*', 'C', 'E', '1', '*', '-', '%', '+', '$', '8', ';', '/', '(', 'A', '&', '#', ',', '%', '"', "'", '@', ';', '/', "'", '!', '6', '?', '-', '5', "'"]

# For unseen time series, you can do 

print("\n\nOn test set")
symbols_test, starts = jabba.transform(X_test)          # use same symbols!
print("new shape:", jabba.new_shape)
X_test_recon = jabba.inverse_transform(symbols_test, starts)

print("error on test set:", np.mean((X_test - X_test_recon)**2))
print(f"Test set reconstructed with {len(jabba.parameters.alphabets)} shared symbols")



# Speedup fABBA
import time
from fABBA import fABBA

data = np.random.randn(20000)

i = 1
start = time.time()
pabba = fABBA(tol=0.05, alpha=0.1, verbose=0, partition=1)
symbols_pabba = pabba.fit_transform(data)
reconstruction_pabba1 = pabba.inverse_transform(symbols_pabba)
end = time.time()
error = np.mean((data - reconstruction_pabba1)**2)
print(f"fABBA with {i} parallel jobs took {end - start:.2f} seconds with error {error:.6f}")

i=5 # 10 jobs
start = time.time()
pabba = fABBA(tol=0.05, alpha=0.1, verbose=0, partition=1)
symbols_pabba = pabba.fit_transform(data)
reconstruction_pabba2 = pabba.inverse_transform(symbols_pabba)
end = time.time()
error = np.mean((data - reconstruction_pabba2)**2)
print(f"fABBA with {i} parallel jobs took {end - start:.2f} seconds with error {error:.6f}")

