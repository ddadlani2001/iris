import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from iris_nn import Model

# Create an instance of our model
model = Model()

url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
my_df = pd.read_csv(url)

# Change last column from sting to int
my_df['species'] = my_df['species'].replace('setosa', 0.0)
my_df['species'] = my_df['species'].replace('versicolor', 1.0)
my_df['species'] = my_df['species'].replace('virginica', 2.0)

# Train Test Split! Set X, y
X = my_df.drop('species', axis=1)
y = my_df['species']

# Convert df to numpy arrays
X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=41)

# Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y labels to tensors long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set the criterion of our model to measure the error, how far off predictions are from the data
criterion = nn.CrossEntropyLoss()

# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs) - we want to lower lr
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train our model
# Epochs? (one run through of the training data in our network)
epochs = 100
losses = []

for i in range(epochs):
    # Go forward and get a prediction
    y_pred = model.forward(X_train)  # Get predicted results

    # Measure the loss/error
    loss = criterion(y_pred, y_train)  # predicted values vs y_train

    # Keep track of our losses
    losses.append(loss.detach().numpy())

    # Print every 10 epochs
    if i % 10 == 0:
        print(f'Epoch {i} and loss: {loss.item()}')

    # Back propagation: feed errors back through network to learn better/more (fine-tune nn weights)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # plt.plot(range(epochs), losses)
    # plt.ylabel("loss/error")
    # plt.xlabel("Epoch")
    # plt.show()

# Evaluate Model on Test Data Set (validate model on test set)
with torch.no_grad(): # Turn off back propagation
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test) # find loss/error

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val= model.forward(data)

        # What type of flower our network thinks it is
        print(f'{i+1}.)  {str(y_val)} \t {y_test[i]}')

        # Check if correct
        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'we got {correct} correct')





