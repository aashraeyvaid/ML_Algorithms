import numpy as np
import pandas as pd

def sigmoid(x):
  s = 1 / (1 + np.exp(-x))
  return s

def logistic_regression(dataset, unseen_dataset):
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

    n_weights=len(dataset.columns)-1
    weights=np.random.rand(n_weights)

    learning_rate=0.01
    epochs=20

    for i in range(1,epochs+1):
        indices = np.random.permutation(len(x_train))
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        for x,y in zip(x_shuffled,y_shuffled):
            z=np.dot(weights,x)
            p=sigmoid(z)
            error=p-y
            weights=weights- learning_rate*error*x
    
    predictions=[]

    for x in x_test:
        z=np.dot(weights,x)
        p=sigmoid(z)

        if p>=0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    predictions=np.array(predictions)

    accuracy=np.sum(predictions==y_test)/len(y_test)
    print('Accuracy:',accuracy)

    unseen_dataset=unseen_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    unseen_dataset = (unseen_dataset - mean) / std
    unseen_dataset = unseen_dataset.values

    unseen_predictions = []

    for x in unseen_dataset:
        z = np.dot(weights, x)
        p = sigmoid(z)

        if p >= 0.5:
            unseen_predictions.append(1)
        else:
            unseen_predictions.append(0)

    return unseen_predictions

