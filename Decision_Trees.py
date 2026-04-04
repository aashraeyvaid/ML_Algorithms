import numpy as np
import pandas as pd

# Helper Functions:
# ===================

def probablity(x, data):
    data=np.array(data)
    if len(data) == 0:
        return 0
    return np.sum(data==x)/len(data)

def accuracy(pred, y):
    return np.sum(pred.to_numpy() == y) / len(y)




# Builder Parameter Functions
# ===========================

def entropy(y):
    if len(y) == 0:
        return 0

    classes = np.unique(y)
    ent = 0
    
    for c in classes:
        p = probablity(c, y)
        if p > 0:
            ent += p * np.log2(p)   # ✅ better to use log2
    
    return -ent


def gini_impurity(y):
    if len(y) == 0:
        return 0

    classes = np.unique(y)
    gi = 0

    for c in classes:
        gi += probablity(c, y) ** 2

    return 1 - gi


def information_gain(x, y, feature, impurity='gini'):
    
    if impurity == 'gini':
        parent_impurity = gini_impurity(y)
        impurity_func = gini_impurity

    elif impurity == 'entropy':
        parent_impurity = entropy(y)
        impurity_func = entropy

    else:
        raise ValueError('Invalid impurity!')

    values = np.unique(x[feature])

    weighted_impurity = 0

    for v in values:
        mask = (x[feature] == v)
        subset_y = y[mask]

        weight = len(subset_y) / len(y)
        weighted_impurity += weight * impurity_func(subset_y)

    return parent_impurity - weighted_impurity




# Tree Builder Function
# =====================

def build_tree(x, y, features, depth, impurity_function='gini'):
    if len(set(y)) == 1:
        return y.iloc[0]
    
    if depth == 0 or len(features) == 0:
        y.mode()[0]
    
    igs = [information_gain(x, y, feature, impurity_function) for feature in features]
    best_feature = features[np.argmax(igs)]

    tree = {best_feature: {}}

    values = np.unique(x[best_feature])

    remaining_features = features.drop(best_feature)

    for v in values:
        subset = x[x[best_feature] == v]
        sub_y = y[subset.index]

        if len(subset) == 0:
            tree[best_feature][v] = y.mode()[0]
        else:
            subtree = build_tree(
                subset.drop(columns=[best_feature]),
                sub_y,
                remaining_features,
                depth - 1,
                impurity_function
            )
            tree[best_feature][v] = subtree

    return tree





# Core Model Function
# ===================

def decision_tree_classifier(dataset, unseen_data, builder='gini', height=5):
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    features = dataset.columns[:-1]

    split = int(0.7 * len(dataset))

    train_data = dataset.iloc[:split]
    test_data = dataset.iloc[split:]
    
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    if builder not in ['gini', 'entropy']:
        raise ValueError('Invalid builder function chosen!')

    decision_tree = build_tree(
        x_train,
        y_train,
        features,
        height,
        builder
    )
    
    test_predictions = x_test.apply(lambda row: predict(decision_tree, row), axis=1)

    print(f'Accuracy: {accuracy(test_predictions, y_test)}')

    if isinstance(unseen_data, pd.Series):
        actual_predictions = predict(decision_tree, unseen_data)
    else:
        actual_predictions = unseen_data.apply(lambda row: predict(decision_tree, row), axis=1)

    return actual_predictions




# Prediction Function
# ===================

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    value = sample[feature]

    if value in tree[feature]:
        return predict(tree[feature][value], sample)
    else:
        return 0  # fallback