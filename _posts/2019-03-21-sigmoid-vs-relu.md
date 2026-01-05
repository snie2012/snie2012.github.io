---
layout: post
title:  Sigmoid vs ReLU
date:   2019-03-21
categories: ml
---
It's well known that ReLU should be used over Sigmoid for activation functions because of gradient vanishing and slow learning. I played a bit with a simple setup to see their effectiveness.

First, let's write a simple MLP with Sigmoid as the activation function:
```python
import numpy as np

class MLP:
    def __init__(self, layer_sizes: list[int]):
      # layer_sizes includes the size of the input layer
      self.layer_sizes = layer_sizes
      self.num_layers = len(self.layer_sizes)
      self.weights = []
      self.biases = []

      for layer in range(1, self.num_layers):
        # the 0.01 multiplier is important to keep numbers within scale in this simple example
        self.weights.append(np.random.rand(self.layer_sizes[layer-1], self.layer_sizes[layer]) * 0.01)
        self.biases.append(np.zeros((1, self.layer_sizes[layer])))

    def sigmoid(self, z):
      return 1 / (1 + np.exp(-z))

    def forward(self, X):
      # X shape: [batch_size, num_features]
      # store activations and values before activations for backpropagation use
      activations = [X]
      pre_activations = []

      for i in range(self.num_layers - 1):
        # @ is matmul
        pre = activations[-1] @ self.weights[i] + self.biases[i]

        pre_activations.append(pre)
        activations.append(self.sigmoid(pre))

      return pre_activations, activations
```

Then, let's add the backpropagation pass:
```python
class MLP:
    def sigmoid_derivative(self, z):
      return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def backward(self, Y, pre_activations, activations, learning_rate, epoch):
      num_samples = Y.shape[0]

      # compute gradient of loss on the output layer of pre_activations
      # assume it's an MSE loss
      dz = 2 * (activations[-1] - Y) * self.sigmoid_derivative(pre_activations[-1])

      for i in range(self.num_layers - 2, -1, -1):
        # shape : (D_(L-1), D_L) = (N, D_(L-1)).T * (N, D_L)
        #    the output shape matches the shape of weights
        dw = (activations[i].T @ dz) / num_samples

        # sum over the first axis, this means sum over all samples, shape is changed from (N, D) to (1, D)
        #  the outcome shape matches the shape of bias
        db = np.sum(dz, axis=0, keepdims=True) / num_samples

        # update dz
        if i > 0:
          # 1. dz is in the front, weight needs to be transposed
          # 2. take the i-1 index of pre_activations (vs i index for activations and weights)
          dz = (dz @ self.weights[i].T) * self.sigmoid_derivative(pre_activations[i-1])

        self.weights[i] -= learning_rate * dw
        self.biases[i] -= learning_rate * db
```

Then let's write the train and predict function:
```python
class MLP:
    def train(self, X, Y, num_epoch, learning_rate):
      for epoch in range(num_epoch):
        pre_activations, activations = self.forward(X)
        self.backward(Y, pre_activations=pre_activations, activations=activations, learning_rate=learning_rate, epoch=epoch)

        if epoch % 100 == 0:
          loss = np.mean((activations[-1] - Y) ** 2)
          print(f"Epoch: {epoch}, loss {loss}")

    def predict(self, X):
      _, activations = self.forward(X)
      return activations[-1]
```

Then, we can create some synthetic data for training and testing:
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(42)
num_samples = 500
input_dim = 8
output_dim = 1

# create data for two classes, one class with label 0 and one class with label 1
X1 = np.random.randn(num_samples // 2, input_dim) + np.array([2] * input_dim)
X2 = np.random.randn(num_samples // 2, input_dim) + np.array([-2] * input_dim)
X = np.vstack([X1, X2])
y = np.vstack([np.ones((num_samples // 2, output_dim)), np.zeros((num_samples // 2, output_dim))])

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Finally, we are ready to initialize and train the model using the following code:
```python
# Define the hidden layers
hidden_dims = []

layer_sizes = [input_dim, *hidden_dims, output_dim]
model = MLP(layer_sizes)
model.train(X_train, y_train, num_epoch=1000, learning_rate=0.1)

# Make predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Evaluate the model
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```

What's interesting is how the loss and test accuracy changes when we change the number of hidden layers. 
Here is what it looks like with an increasing number of hidden layers:
```python
hidden_dims = [16]
# Epoch: 0, loss 0.24969308533734108
# Epoch: 100, loss 0.24635126435577548
# Epoch: 200, loss 0.1933330774724225
# Epoch: 300, loss 0.04350553054137293
# Epoch: 400, loss 0.014043079342073987
# Epoch: 500, loss 0.0074231675264761696
# Epoch: 600, loss 0.00484563232437703
# Epoch: 700, loss 0.00353025839318329
# Epoch: 800, loss 0.0027479963340805093
# Epoch: 900, loss 0.0022351556503922075
# Test Accuracy: 1.00

hidden_dims = [16, 32]
# Epoch: 0, loss 0.24998323021385307
# Epoch: 100, loss 0.24989505411105845
# Epoch: 200, loss 0.24989428917701395
# Epoch: 300, loss 0.2498933984651159
# Epoch: 400, loss 0.24989235256675568
# Epoch: 500, loss 0.2498911126725463
# Epoch: 600, loss 0.24988962663101855
# Epoch: 700, loss 0.2498878229778187
# Epoch: 800, loss 0.24988560165599766
# Epoch: 900, loss 0.2498828191733827
# Test Accuracy: 0.46

hidden_dims = [16, 32, 16]
# Epoch: 0, loss 0.24990007245522378
# Epoch: 100, loss 0.24989987040500722
# Epoch: 200, loss 0.24989986814116372
# Epoch: 300, loss 0.2498998658547434
# Epoch: 400, loss 0.24989986354503468
# Epoch: 500, loss 0.2498998612115275
# Epoch: 600, loss 0.2498998588537019
# Epoch: 700, loss 0.24989985647102753
# Epoch: 800, loss 0.24989985406296378
# Epoch: 900, loss 0.24989985162895892
# Test Accuracy: 0.46
```

We can see that Sigmoid performs perfectly with a single hidden layer, but it can achieve no better than random guessing with two or more layers. Let's now replace the activation function with ReLU:
```python
class MLP:
    def relu(self, x):
      return np.maximum(0, x)

    def relu_derivative(self, z):
      grad = np.zeros_like(z)
      grad[z > 0] = 1
      return grad
    
    # replace self.sigmoid with self.relu in the forward pass
    # replace self.sigmoid_derivative with self.relu_derivative in the backward pass
```

With the ReLU activation function, this is what we see as the loss and test accuracy as we change the number of hidden layers:
```python
hidden_dims = [16]
# Epoch: 0, loss 0.5072307176440325
# Epoch: 100, loss 0.008881138883477696
# Epoch: 200, loss 0.007414984359001941
# Epoch: 300, loss 0.006123937988799423
# Epoch: 400, loss 0.0049971759648809125
# Epoch: 500, loss 0.004472758108475496
# Epoch: 600, loss 0.004412912198861297
# Epoch: 700, loss 0.004408119968091293
# Epoch: 800, loss 0.0044075127091029774
# Epoch: 900, loss 0.00440738866703719
# Test Accuracy: 1.00

hidden_dims = [16, 32]
# Epoch: 0, loss 0.5095474480894288
# Epoch: 100, loss 0.13862116800663518
# Epoch: 200, loss 0.009042771273083141
# Epoch: 300, loss 0.00793566048724749
# Epoch: 400, loss 0.006933118582207134
# Epoch: 500, loss 0.005962305748896169
# Epoch: 600, loss 0.005055722890355142
# Epoch: 700, loss 0.0045220929578166915
# Epoch: 800, loss 0.004420407538998189
# Epoch: 900, loss 0.004409597584398879
# Test Accuracy: 1.00

hidden_dims = [16, 32, 16]
# Epoch: 0, loss 0.5099626260455881
# Epoch: 100, loss 0.2498654237449173
# Epoch: 200, loss 0.2498436800487159
# Epoch: 300, loss 0.24980429157623468
# Epoch: 400, loss 0.2497204720653153
# Epoch: 500, loss 0.24948757632299043
# Epoch: 600, loss 0.24833190070710132
# Epoch: 700, loss 0.1513366573565615
# Epoch: 800, loss 0.009248190634186335
# Epoch: 900, loss 0.00825585784190759
# Test Accuracy: 1.00

hidden_dims = [16, 32, 16, 8]
# Epoch: 0, loss 0.5099986751947574
# Epoch: 100, loss 0.24989924414480175
# Epoch: 200, loss 0.24989921448963798
# Epoch: 300, loss 0.2498991841176106
# Epoch: 400, loss 0.2498991529914859
# Epoch: 500, loss 0.24989912107217216
# Epoch: 600, loss 0.2498990883185868
# Epoch: 700, loss 0.24989905468751558
# Epoch: 800, loss 0.2498990201334597
# Epoch: 900, loss 0.24989898460847
# Test Accuracy: 0.46
```

As we can see, comparing to Sigmoid, ReLU can achieve perfect accuracy with up to 3 layers easily! Meanwhile, we can also see that the loss decrease becomes slower as the number of layers increases and it failed to converge with four hidden layers.

It feels magical to see, even with such simple MLPs, that ReLU makes a big difference compared to Sigmoid. It would be very interesting to further explore the dynamics of the gradients from layer to layer, to inspect when and where they plateaued.

Furthermore, this sparks a much wider range of questions on the design choices of modern deep neural nets, and how each of the following choices would make a difference in their training dynamics:
- Weight initialization techniques (Xavier/Kaming initialization). Even in this very simple example, ReLU would not converge if the weights are not multiplied by 0.01.
- Activation functions. There are many more to explore than ReLU, for example LeakyReLU, tanh, GeLU etc.
- Skip connection / Residual connection. This is meant to stabilize deep nets.
- Optimizer choice. The optimizer is a barebone SGD in this case. What would happen if we replace it with momentum or Adam or AdamW?
- Layer/batch normalization.
- Type of loss. Would it make a difference to replace MSE with CE in this simple case?
- Learning rate schedule and the effect of number of epochs. For example linear warmup + cosine decay.

All the above design choices can be coupled with any of the following setups:
- Input data complexity.
- Network architecture: MLP vs CNN vs RNN vs LSTM vs Transformer etc.
- Same architecture with wider layers.
- Same architecture with deeper layers.

It is amazingly simple to see how all these designs and innovations come together to enable us to train very deep neural nets on very complex data effectively, and how a very simple MLP + synthetic data can reflect the effectiveness of some of these design choices.

