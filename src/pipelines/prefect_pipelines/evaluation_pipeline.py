from prefect import task,flow
from prefect.task_runners import ConcurrentTaskRunner,DaskTaskRunner
from configurations import model_metrics,business_metrics,X_train_y_train


@task
def load_data():
    """
    Component for loading the  X_test and y_test which will be used
    for model evaluation pipeline
    """
    X_test,y_test=X_train_y_train(path="data/test_df.csv")
    return X_test,y_test

@task
def load_model(model_path):
    """
    Component for loading the model trained after we saved them in the train pipelne:
    """
    import joblib

    model=joblib.load(model_path)

    return model


@task
def model_evaluation(X_test,y_test,model):
    y_pred=model.predict(X_test)

    model_metrics_eval=model_metrics(y_test=y_test,X_test=X_test,model=model)

    return model_metrics_eval


@task
def model_business_evaluation(X_test,y_test,model):

    from configurations import curve_treats,business_metrics
    y_true=model.predict(X_test)

    thresholds,treat_none,treat_all,nets=curve_treats(y_true=y_test,X_test=X_test,model=model)

    model_business_metrics=business_metrics(thresholds,treat_none,treat_all,nets)

    return model_business_metrics


@task
def save_metrics(metrics,path):
    import joblib

    joblib.dump(metrics,path)

@flow
def model_evaluation_pipeline():

    X_test,y_test=load_data()
    blue_model=load_model("models/blue_model.pkl")
    green_model=load_model("models/green_model.pkl")

    blue_model_metrics=model_evaluation(X_test=X_test,y_test=y_test,model=blue_model)
    green_model_metrics=model_evaluation(X_test=X_test,y_test=y_test,model=green_model)


    blue_model_business_metrics=model_business_evaluation(X_test=X_test,y_test=y_test,model=blue_model)
    green_model_business_metrics=model_business_evaluation(X_test=X_test,y_test=y_test,model=green_model)

    save_metrics(blue_model_metrics,"metrics/model_metrics/blue_model_metrics.pkl")
    save_metrics(green_model_metrics,"metrics/model_metrics/green_model_metrics.pkl")

    
    save_metrics(blue_model_business_metrics,"metrics/business_metrics/blue_model_business_metrics.pkl")
    save_metrics(green_model_business_metrics,"metrics/business_metrics/green_model_business_metrics.pkl")

    


    


