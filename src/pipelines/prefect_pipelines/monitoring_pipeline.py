from prefect import task,flow


import numpy as np
import pandas as pd

np.NInf=np.inf
np.Inf=np.inf

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import LabelDrift,FeatureDrift

@task
def load_data():
    

    train_df=pd.read_csv("data/train_df.csv")
    test_df=pd.read_csv("data/test_df.csv")

    return train_df,test_df

@task
def tabulate_data(train_data,test_data):
    train_set=Dataset(train_data,cat_features=[],label="loan_status")
    test_set=Dataset(test_data,cat_features=[],label="loan_status")

    return train_set,test_set

@task
def feature_drift_scores(train_data,test_data,train_df):

    ## get the feature drift:
    drift=FeatureDrift()

    ## run the drift :
    results=drift.run(train_data,test_data)

    ## get tbe feature columns:
    X_train=train_df.drop("loan_status",axis=1)
    features=X_train.columns

    ## get drift scores
    feature_drift_scores=[]

    for feature in features:

        drift_score=results.value[feature]['Drift score']

        feature_drift_scores.append(drift_score.item())
    
    ## save as dataframe:
    drift_scores_df=pd.DataFrame(

        {
    "feature":features,

    "drift_score":feature_drift_scores
    }

    )

    return drift_scores_df
    ## save the drift scores dataframe:
    
def save_scores(drift_scores_data):
    drift_scores_data.to_csv("data/drift_data.csv",index=False)


@flow
def monitoring_features():
    train_df,test_df=load_data()

    train_data,test_data=tabulate_data(train_data=train_df,test_data=test_df)

    drift_scores=feature_drift_scores(train_data=train_data,test_data=test_data,train_df=train_df)

    save_scores(drift_scores_data=drift_scores)


