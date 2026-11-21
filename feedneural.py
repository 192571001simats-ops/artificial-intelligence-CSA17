from math import exp

def sigmoid(x):
    return 1 / (1 + exp(-x))

x1 = float(input("Enter first input: "))
x2 = float(input("Enter second input: "))

h1 = sigmoid(x1 * 0.5 + x2 * 0.2 + 0.1)
h2 = sigmoid(x1 * 0.3 + x2 * 0.7 + 0.1)

output = sigmoid(h1 * 0.6 + h2 * 0.4 + 0.1)

print("Neural network output:", round(output, 4))
