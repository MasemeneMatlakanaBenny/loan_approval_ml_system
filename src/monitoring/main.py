import numpy as np
import pandas as pd

np.NInf=np.inf
np.Inf=np.inf

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import LabelDrift,FeatureDrift

np.NInf=np.inf
np.Inf=np.inf


train_df=pd.read_csv("test_df.csv")

test_df=pd.read_csv("train_df.csv")

train_data=Dataset(train_df,cat_features=[],label="loan_status")
test_data=Dataset(test_df,cat_features=[],label="loan_status")

drift=FeatureDrift()

results=drift.run(train_data,test_data)



X_train=train_df.drop("loan_status",axis=1)
features=X_train.columns
feature_drift_scores=[]

for feature in features:
    drift_score=results.value[feature]['Drift score']
    feature_drift_scores.append(drift_score.item())

drift_scores_df=pd.DataFrame({
    "feature":features,
    "drift_score":feature_drift_scores})


drift_scores_df.to_csv("data/drift_data.csv",index=False)
