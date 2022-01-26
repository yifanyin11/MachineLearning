import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# fetch data and scaling
X, y = fetch_openml('mnist_784', return_X_y=True)
X = X.values
scaler = StandardScaler()
scaler.fit(X)
X_s = scaler.transform(X)

# perform PCA 

n = np.size(X_s, axis=1)
pca = PCA(n_components=0.95, svd_solver='full')
pca.fit(X_s)
evr = pca.explained_variance_ratio_
global cvr
cvr = np.cumsum(evr)


def sorted_key(par):
    global cvr
    return np.absolute(cvr[par] - 0.95)


plt.figure(figsize=(8, 6))
plt.plot(cvr)
plt.xlabel('Dimensions')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance of PCA')

tol = 10 ** -3
candidates = np.where(np.absolute(cvr - 0.95) < tol)[0]
candidates = sorted(candidates, key=sorted_key)
dim = candidates[0]

print('The dimension corresponding to preserving 95% cumulative variance is', dim, '.')

plt.plot(dim, cvr[dim], marker='o')
plt.annotate('(%s, %s)' % ((dim), (np.round(cvr[dim], 2))), xy=(dim, cvr[dim]))
plt.show()

tol = 10 ** -4
print('The dimension corresponding to preserving 95% cumulative variance is',
      np.where(np.absolute(cvr - 0.95) < tol)[0][0], '.')

X_new = pca.transform(X)
X_back = pca.inverse_transform(X_new)

sel = np.random.randint(np.size(X_back, axis=0) + 1)
img_before = np.reshape(X[sel, :], (28, 28))
img_after = np.reshape(X_back[sel, :], (28, 28))
fg = plt.figure(figsize=(10, 4))
ax1 = fg.add_subplot(121)
ax2 = fg.add_subplot(122)
ax1.imshow(img_before, cmap='gray')
ax1.set_title('A image before compression')

ax2.imshow(img_after, cmap='gray')
ax2.set_title('The image after compression')
plt.show()
