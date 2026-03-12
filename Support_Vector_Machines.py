import numpy as np
import pandas as pd

def support_vector_machines_classifier(dataset,unseen_dataset):
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    split=int(0.7*len(dataset))

    train_data=dataset.iloc[:split]
    test_data=dataset.iloc[split:]

    x_train=train_data.iloc[:,:-1]
    y_train=train_data.iloc[:,-1].values

    x_test=test_data.iloc[:,:-1]
    y_test=test_data.iloc[:,-1].values

    mean=x_train.mean()
    std=x_train.std()

    x_train=(x_train-mean)/std
    x_test=(x_test-mean)/std

    x_train=x_train.values
    x_test=x_test.values

    y_train = np.where(y_train==0,-1,1)
    y_test = np.where(y_test==0,-1,1)

    n_weights=len(dataset.columns)-1
    weights = np.random.randn(n_weights) * 0.01
    bias=np.random.rand()

    learning_rate=0.01
    lamda=0.01
    epochs=500

    for epoch in range(1,epochs+1):
        indices = np.random.permutation(len(x_train))
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        for x,y in zip(x_shuffled,y_shuffled):
            z = (np.dot(weights,x) + bias) * y

            if z>=1:
                weights = weights - learning_rate * 2 * lamda * weights

            else:
                weights = weights - learning_rate * (-y * x + 2 * lamda * weights)
                bias = bias + learning_rate * y
        
        if epoch % 50 == 0:
            z = y_train * (np.dot(x_train, weights) + bias)
            loss = np.mean(np.maximum(0, 1 - z))
            print(f'Epoch: {epoch}, Loss: {loss}')
        
    predictions = np.sign(np.dot(x_test,weights) + bias)

    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f'\nAccuracy: {accuracy}', end='\n\n')

    unseen_dataset = (unseen_dataset - mean) / std
    unseen_dataset = unseen_dataset.values

    unseen_predictions = np.sign(np.dot(unseen_dataset,weights) + bias)

    return unseen_predictions
