import numpy as np
import pandas as pd

# Distances
def _validate_vectors(p1, p2):
    """
    Internal helper to validate and convert inputs to numpy arrays.
    """
    if p1 is None or p2 is None:
        raise ValueError("Input vectors cannot be None.")

    p1 = np.array(p1)
    p2 = np.array(p2)

    if p1.ndim != 1 or p2.ndim != 1:
        raise ValueError("Inputs must be 1-D vectors.")

    if len(p1) == 0 or len(p2) == 0:
        raise ValueError("Vectors cannot be empty.")

    if len(p1) != len(p2):
        raise ValueError("Vectors must be of same length.")

    return p1, p2


def euclidean_distance(p1, p2):
    p1, p2 = _validate_vectors(p1, p2)
    return np.sqrt(np.sum((p1 - p2) ** 2))


def manhattan_distance(p1, p2):
    p1, p2 = _validate_vectors(p1, p2)
    return np.sum(np.abs(p1 - p2))


def hamming_distance(s1, s2):
    s1, s2 = _validate_vectors(s1, s2)
    return np.sum(s1 != s2)


def cosine_distance(p1, p2):
    p1, p2 = _validate_vectors(p1, p2)

    dot_product = np.dot(p1, p2)
    norm_p1 = np.linalg.norm(p1)
    norm_p2 = np.linalg.norm(p2)

    if norm_p1 == 0 or norm_p2 == 0:
        raise ValueError("Cosine distance undefined for zero vector.")

    cosine_similarity = dot_product / (norm_p1 * norm_p2)

    return 1 - cosine_similarity

def _validate_vectors(p1, p2):
    """
    Internal helper to validate and convert inputs to numpy arrays.
    """
    if p1 is None or p2 is None:
        raise ValueError("Input vectors cannot be None.")

    p1 = np.array(p1)
    p2 = np.array(p2)

    if p1.ndim != 1 or p2.ndim != 1:
        raise ValueError("Inputs must be 1-D vectors.")

    if len(p1) == 0 or len(p2) == 0:
        raise ValueError("Vectors cannot be empty.")

    if len(p1) != len(p2):
        raise ValueError("Vectors must be of same length.")

    return p1, p2

def euclidean_distance(p1, p2):
    p1, p2 = _validate_vectors(p1, p2)
    return np.sqrt(np.sum((p1 - p2) ** 2))


def manhattan_distance(p1, p2):
    p1, p2 = _validate_vectors(p1, p2)
    return np.sum(np.abs(p1 - p2))


def hamming_distance(s1, s2):
    s1, s2 = _validate_vectors(s1, s2)
    return np.sum(s1 != s2)


def cosine_distance(p1, p2):
    p1, p2 = _validate_vectors(p1, p2)

    dot_product = np.dot(p1, p2)
    norm_p1 = np.linalg.norm(p1)
    norm_p2 = np.linalg.norm(p2)

    if norm_p1 == 0 or norm_p2 == 0:
        raise ValueError("Cosine distance undefined for zero vector.")

    cosine_similarity = dot_product / (norm_p1 * norm_p2)

    return 1 - cosine_similarity


def accuracy(predictions,y_test):
    count=0
    for i,j in zip(predictions,y_test):
        if i==j:
            count+=1

    accuracy=count/len(y_test)

    return accuracy


# Normal KNN Function
def knn(dataset,tester_data,k=3):
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    split_index=int(0.7*len(dataset))

    train_dataset=dataset.iloc[:split_index]
    test_dataset=dataset.iloc[split_index:]

    x_train=train_dataset.iloc[:,:-1]
    y_train=train_dataset.iloc[:,-1].values

    x_test=test_dataset.iloc[:,:-1]
    y_test=test_dataset.iloc[:,-1].values

    mean=x_train.mean()
    std=x_train.std()

    x_train=(x_train-mean)/std
    x_test=(x_test-mean)/std

    x_train=x_train.values
    x_test=x_test.values

    predictions=[]

    for test_point in x_test:
        distances=[]

        for i, train_point in enumerate(x_train):
            dist=euclidean_distance(test_point,train_point)
            distances.append((dist,y_train[i]))

        distances.sort(key=lambda x: x[0])

        k_nearest=distances[:k]

        labels=[j for i,j in k_nearest]
        prediction=max(set(labels), key=labels.count)

        predictions.append(prediction)

    acc=accuracy(predictions,y_test)

    print(f'Accuracy of model: {acc} for value of k: {k}')

    actual_point_predictions=[]

    tester_data = np.array(tester_data)

    if tester_data.ndim == 1:
        tester_data = tester_data.reshape(1, -1)

    tester_data=(tester_data-mean.values)/std.values

    for actual_point in tester_data:

        actual_point_distances=[]

        for i, train_point in enumerate(x_train):
            actual_point_dist=euclidean_distance(train_point,actual_point)
            actual_point_distances.append((actual_point_dist,y_train[i]))

        actual_point_distances.sort(key=lambda x:x[0])

        actual_point_k_nearest=actual_point_distances[:k]

        actual_point_labels=[j for i,j in actual_point_k_nearest]
        actual_point_prediction=max(set(actual_point_labels), key=actual_point_labels.count)

        actual_point_predictions.append(actual_point_prediction)

    return actual_point_predictions


# Cross Validation KNN Function

def k_fold_cross_validation(dataset,tester_data,k_neighbors=3,k_folds=5):
    dataset=dataset.sample(frac=1,random_state=42).reset_index(drop=True)

    split_index=int(0.8*len(dataset))

    unseen_dataset=dataset.iloc[split_index:]
    dataset=dataset.iloc[:split_index]

    fold_size=len(dataset)//k_folds
    
    mean_accuracy_scores=[]

    for k in range(1,k_neighbors+1):
        fold_accuracies=[]
        for fold in range(k_folds):
            start=fold*fold_size
            end=start+fold_size

            test_data=dataset.iloc[start:end]
            train_data=pd.concat([dataset.iloc[:start],dataset.iloc[end:]])

            x_train=train_data.iloc[:,:-1]
            y_train=train_data.iloc[:,-1].values

            x_test=test_data.iloc[:,:-1]
            y_test=test_data.iloc[:,-1]

            mean=x_train.mean()
            std=x_train.std()

            x_train=(x_train-mean)/std
            x_test=(x_test-mean)/std

            x_train=x_train.values
            x_test=x_test.values

            predictions=[]

            for test_point in x_test:
                distances=[]

                for i, train_point in enumerate(x_train):
                    dist=euclidean_distance(train_point,test_point)
                    distances.append((dist,y_train[i]))

                distances.sort(key=lambda x:x[0])

                k_nearest=distances[:k]

                labels=[label for _, label in k_nearest]
                prediction=max(set(labels),key=labels.count)

                predictions.append(prediction)
            
            acc=accuracy(predictions, y_test)
            fold_accuracies.append(acc)

            print(f'K= {k}, K_Fold= {fold}, Accuracy= {acc}')
        
        mean_accuracy=np.mean(fold_accuracies)
        mean_accuracy_scores.append({'K':k,'Mean_Accuracy':mean_accuracy})
        print(f'K= {k} Mean Accuracy= {mean_accuracy}')

    mean_accuracy_scores.sort(key=lambda x: x['Mean_Accuracy'], reverse=True)

    print(f'\nBest K: {mean_accuracy_scores[0]}')

    best_k = mean_accuracy_scores[0]['Mean_Accuracy']
    best_k_value = mean_accuracy_scores[0]['K']

    print(f'\nBest K selected: {best_k_value}')

    train_full = dataset
    test_full = unseen_dataset

    x_train = train_full.iloc[:, :-1]
    y_train = train_full.iloc[:, -1].values

    x_test = test_full.iloc[:, :-1]
    y_test = test_full.iloc[:, -1].values

    mean = x_train.mean()
    std = x_train.std()

    x_train = ((x_train - mean) / std).values
    x_test = ((x_test - mean) / std).values

    predictions = []

    for test_point in x_test:
        distances = []

        for i, train_point in enumerate(x_train):
            dist = euclidean_distance(train_point, test_point)
            distances.append((dist, y_train[i]))

        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:best_k_value]

        labels = [label for _, label in k_nearest]
        prediction = max(set(labels), key=labels.count)

        predictions.append(prediction)

    final_test_accuracy = np.mean(np.array(predictions) == y_test)

    print(f'Final Unseen Test Accuracy: {final_test_accuracy}')

    tester_data = np.array(tester_data)

    if tester_data.ndim == 1:
        tester_data = tester_data.reshape(1, -1)

    tester_data = (tester_data - mean.values) / std.values

    custom_predictions = []

    for actual_point in tester_data:
        distances = []

        for i, train_point in enumerate(x_train):
            dist = euclidean_distance(train_point, actual_point)
            distances.append((dist, y_train[i]))

        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:best_k_value]

        labels = [label for _, label in k_nearest]
        prediction = max(set(labels), key=labels.count)

        custom_predictions.append(prediction)

    return custom_predictions

