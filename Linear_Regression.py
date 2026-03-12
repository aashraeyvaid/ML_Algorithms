def linear_regression(dataset,unseen_dataset):
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
    bias=np.random.rand()

    learning_rate=0.01
    epochs=500

    for i in range(1,epochs+1):
        indices = np.random.permutation(len(x_train))
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        for x,y in zip(x_shuffled,y_shuffled):
            y_cap=np.dot(weights,x) + bias
            error=y-y_cap
            bias=bias + learning_rate * error
            weights=weights + learning_rate * error * x

        if i % 50 == 0:
            loss = np.mean((np.dot(x_train,weights)+bias - y_train)**2)
            print("Epoch:",i,"Loss:",loss)
    
    predictions = np.dot(x_test, weights) + bias

    mse = np.mean((predictions - y_test)**2)
    print("\n\033[1mMSE:", mse,end='\n\n')

    unseen_dataset=unseen_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    unseen_dataset = (unseen_dataset - mean) / std
    unseen_dataset = unseen_dataset.values

    unseen_predictions = np.dot(unseen_dataset,weights) + bias

    return unseen_predictions