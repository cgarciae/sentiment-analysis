#%%
import numpy as np

#%%
a = np.random.uniform(size=(10, 22, 64))
b = np.random.uniform(size=(64, 8, 32))

print(a.shape)
print(b.shape)
# %%
x = np.einsum("BLC , CHD -> BLHD", a, b)

print(x.shape)

a1 = a
b1 = b.reshape((64, -1))

y = np.dot(a1, b1).reshape((10, 22, 8, 32))

print(y.shape)

assert np.allclose(x, y)

# %%
np.einsum("b , bc -> bc", np.einsum("ab -> b", a), b)  # %%

# %%
