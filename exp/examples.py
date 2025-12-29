from fABBA import JABBA
import numpy as np
import matplotlib.pyplot as plt

# Simulate 20 independent univariate series
np.random.seed(0)
data = np.cumsum(np.random.randn(20, 800), axis=1)  # random walks

jabba = JABBA(tol=0.2, init='agg', verbose=1)  # auto-digitization
symbols = jabba.fit_transform(data)
recon = jabba.inverse_transform(symbols)


# Get embedding for the each symbol, and concatenate embeddings for the first time series
alphabets = jabba.parameters.alphabets.tolist()
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



# you can load the parameters of ABBA:

print(jabba.parameters)
X_train = np.random.randn(100, 5, 200)
X_test  = np.random.randn(30, 5, 200)


print("On train set")
jabba = JABBA(tol=0.05).fit(X_train)                    # learn vocabulary
print("new shape:", jabba.new_shape)
JABBA(tol=0.05).fit_transform(X_train) 
symbols_train, starts = jabba.transform(X_train)          # use same symbols!
X_train_recon = jabba.inverse_transform(symbols_train, starts)
print("error on train set:", np.mean((X_train - X_train_recon)**2))


print("\n\nOn test set")
symbols_test, starts = jabba.transform(X_test)          # use same symbols!
print("new shape:", jabba.new_shape)
X_test_recon = jabba.inverse_transform(symbols_test, starts)

print("error on test set:", np.mean((X_test - X_test_recon)**2))
print(f"Test set reconstructed with {len(jabba.parameters.alphabets)} shared symbols")

# for unseen time series, you can do 