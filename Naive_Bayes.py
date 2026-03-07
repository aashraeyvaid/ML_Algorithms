import numpy as np
import pandas as pd

def naive_bayes(dataset,unseen_data):
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    split=int(0.7*len(dataset))

    train_dataset=dataset.iloc[:split,:]
    test_dataset=dataset.iloc[split:,:-1]
    y_test=dataset.iloc[split:,-1]

    features=train_dataset.columns[:-1]
    target=train_dataset.columns[-1]


    likelihoods={}
    for feature in features:
        table=pd.crosstab(
            train_dataset[feature],
            train_dataset[target],
            normalize='columns'
        )

        likelihoods[feature]={}

        for cls in table.columns:
            likelihoods[feature][cls] = table[cls].to_dict()

    
    priors=train_dataset[target].value_counts(normalize=True).to_dict()

    test_predictions=[]
    for _, row in test_dataset.iterrows():
        row=row.to_dict()
        
        test_scores={}
        for cls in priors:
            test_score=np.log(priors[cls])
            for ft, value in row.items():
                test_score+=np.log(likelihoods[ft][cls].get(value,1e-6))

            test_scores[cls]=test_score
        test_predictions.append(max(test_scores,key=test_scores.get))

    count=0
    for i,j in zip(y_test.values,test_predictions):
        if i==j:
            count+=1
    accuracy=count/len(test_dataset)

    print(f'Model Accuracy: {accuracy}')


    predictions=[]
    for _, row in unseen_data.iterrows():
        row=row.to_dict()
        
        scores={}
        for cls in priors:
            score=np.log(priors[cls])
            for ft, value in row.items():
                score+=np.log(likelihoods[ft][cls].get(value,1e-6))

            scores[cls]=score
        predictions.append((max(scores,key=scores.get),scores))

    return predictions
