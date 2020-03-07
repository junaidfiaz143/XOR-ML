import numpy as np

# first column is bias
x = np.array([
			[1, 1, 0],
			[1, 0, 1],
			[1, 0, 0],
			[1, 1, 1]])

y = np.array([[1], [1], [0], [0]])

# hyparameters
w1 = np.random.randn(3, 5)
w2 = np.random.randn(6, 1)
lr = 0.09
costs = []
epochs = 5000

m = len(x)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_d(x):
	return sigmoid(x) * (1 - sigmoid(x))

# 1 hidden layer
def f(x, w1, w2):
	a1 = np.matmul(x, w1)
	z1 = sigmoid(a1)

	bias = np.ones((len(z1), 1))
	z1 = np.concatenate((bias, z1), axis=1)

	a2 = np.matmul(z1, w2)
	z2 = sigmoid(a2)

	return a1, z1, a2, z2

def b(a1, x, z1, z2, y):
	delta2 = z2 - y
	Delta2 = np.matmul(z1.T, delta2)
	delta1 = delta2.dot(w2[1:,:].T) * sigmoid_d(a1)
	Delta1 = np.matmul(x.T, delta1)

	return delta2, Delta1, Delta2

# training loop
for i in range(epochs):
	a1, z1, a2, z2 = f(x, w1, w2)

	delta2, Delta1, Delta2 = b(a1, x, z1, z2, y)

	w1 -= lr*(1/m)*Delta1
	w2 -= lr*(1/m)*Delta2

	c = np.mean(np.abs(delta2))
	costs.append(c)

	if i%1000 is 0:
		print(f"Iteration: {i}. Error: {c}")

print("\nTRAINING COMPLETE\n")

def predict(x, w1, w2):
	a1 = np.matmul(x, w1)
	z1 = sigmoid(a1)

	bias = np.ones((len(z1), 1))
	z1 = np.concatenate((bias, z1), axis=1)

	a2 = np.matmul(z1, w2)
	z2 = sigmoid(a2)

	return z2

# predictions
print(f"W1 {w1}")
print(f"W2 {w2}")
print(np.round(predict(x, w1, w2)))