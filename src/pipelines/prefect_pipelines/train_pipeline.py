from prefect import task,flow
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

@task
def load_data():
    """
    Component of the pipeline that will be used to load the X_train and y_train:
    """
    from configurations import X_train_y_train

    X_train,y_train=X_train_y_train()

    return X_train,y_train

@task
def blue_model_dev(X_train,y_train)->LogisticRegression:
    """
    Component of the train pipeline that will be used to train the first model-> blue
    model in this case. The component takes in the X_train and y_train
    """
    model=LogisticRegression(solver="liblinear")

    model.fit(X_train,y_train)

    return model

@task
def green_model_dev(X_train,y_train)->DecisionTreeClassifier:
    """
    Component of the train pipeline that will be used to train the second model ->
    green model in this case.The component takes in the X_train and y_train just like the
    blue model development task

    """
    model=DecisionTreeClassifier(criterion="gini",min_samples_split=3,splitter="best")

    model.fit(X_train,y_train)
    return model

@task
def save_models(model,model_path):
    import joblib

    joblib.dump(model,model_path)


def concurrent_model_dev_pipeline():
    X_train,y_train=load_data()
    blue_model=blue_model_dev(X_train,y_train)
    green_model=green_model_dev(X_train,y_train)

    save_models(blue_model,"models/blue_model.pkl")
    save_models(green_model,"models/green_model.pkl")

if __name__=="__main__":
    concurrent_model_dev_pipeline()





