import numpy as np
import pandas as pd

def rbf_kernel(xi, xj, sigma=1):
    distance_squared = np.linalg.norm(xi - xj)**2
    return np.exp(-distance_squared / (2 * sigma**2))

def compute_kernel_matrix(X, kernel):
    n = X.shape[0]
    K = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            K[i,j] = kernel(X[i], X[j])

    return K

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

def kernel_svm(dataset, unseen_data,kernel):
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

    n_samples = x_train.shape[0]
    alphas = np.zeros(n_samples)

    bias=np.random.rand()

    learning_rate=0.01
    lamda=0.01
    epochs=500
    K = compute_kernel_matrix(x_train, kernel)

    for epoch in range(1,epochs+1):
        indices = np.random.permutation(len(x_train))

        c=1/lamda

        for i in indices:
            prediction_i=np.sum(alphas * y_train * K[:,i])
            gradient = 1 - y_train[i] * prediction_i
            alphas[i] += learning_rate * gradient
            alphas[i]=np.clip(alphas[i],0,c)
        
    support_vectors = np.where(alphas > 1e-5)[0]

    bias_values = []

    for s in support_vectors:
        value = y_train[s] - np.sum(alphas * y_train * K[:, s])
        bias_values.append(value)

    bias = np.mean(bias_values)

    predictions = []

    for x in x_test:
        
        value = 0
        
        for i in range(n_samples):
            if alphas[i] > 1e-5:
                value += alphas[i] * y_train[i] * kernel(x_train[i], x)
        
        value += bias
        
        predictions.append(np.sign(value))

    predictions = np.array(predictions)

    accuracy = np.mean(predictions == y_test)
    print("Accuracy:", accuracy)

    unseen_predictions=[]

    unseen_data = unseen_data.values

    for x in unseen_data:
        
        value = 0
        
        for i in range(n_samples):
            if alphas[i] > 1e-5:
                value += alphas[i] * y_train[i] * kernel(x_train[i], x)
        
        value += bias
        
        unseen_predictions.append(np.sign(value))

    unseen_predictions = np.array(unseen_predictions)

    return unseen_predictions
