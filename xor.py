import numpy as np

x = np.array([
			[1, 0],
			[0, 1],
			[0, 0],
			[1, 1]])

y = np.array([[1], [1], [0], [0]])

# hyparameters
w1 = np.random.randn(2, 4)
w2 = np.random.randn(4, 1)
lr = 0.09
costs = []
epochs = 10000

m = len(x)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_d(x):
	return sigmoid(x) * (1 - sigmoid(x))

# 1 hidden layer
def forward(x, w1, w2):
	a1 = np.matmul(x, w1)
	z1 = sigmoid(a1)

	a2 = np.matmul(z1, w2)
	z2 = sigmoid(a2)

	return a1, z1, a2, z2

def backpropagation(a1, x, z1, z2, y):
	delta2 = z2 - y
	Delta2 = np.matmul(z1.T, delta2)

	delta1 = np.matmul(delta2, w2.T) * sigmoid_d(a1)
	Delta1 = np.matmul(x.T, delta1)

	return delta2, Delta1, Delta2

# training loop
for i in range(epochs):
	a1, z1, a2, z2 = forward(x, w1, w2)

	delta2, Delta1, Delta2 = backpropagation(a1, x, z1, z2, y)

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

	a2 = np.matmul(z1, w2)
	z2 = sigmoid(a2)

	return z2

rearrange_x = np.array([[1, 1],[0, 0], [1, 0], [0, 1]])

print("REARRANGE XOR INPUT VALUES")

print("PREDICTIONS")
predictions = np.round(predict(rearrange_x, w1, w2))

for i in range(rearrange_x.shape[0]):
	print(rearrange_x[i], "->", predictions[i])

print("ORIGINAL XOR INPUT VALUES")

print("PREDICTIONS")
predictions = np.round(predict(x, w1, w2))

for i in range(x.shape[0]):
	print(x[i], "->", predictions[i])