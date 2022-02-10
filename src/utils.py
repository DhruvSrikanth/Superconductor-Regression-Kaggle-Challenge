from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error

def split_data(df):
    '''
    Split data into training and validation sets.
    '''
    X = df.iloc[:, :-1] # get predictors
    y = df.iloc[:, -1] # get target
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=0) # split data in 80% training: 20% validation 
    return X_train, X_validation, y_train, y_validation


def normalize(X_train, X_validation):
    '''
    Normalize training and validation predictor vectors.
    '''
    scaler = StandardScaler()
    scaler.fit(X_train) # fit normalizer to training data
    X_train = scaler.transform(X_train) # normalize training data
    X_validation = scaler.transform(X_validation) # normalize validation data
    return X_train, X_validation


def dimensionality_reduction(X_train, X_validation, n_components=0.90):
    '''
    Perform principal component analysis for dimensionality reduction.
    '''
    pca = PCA(n_components)
    pca.fit(X_train) # fit PCA to training predictor vectors
    X_train = pca.transform(X_train) # perform PCA on training predictor vectors
    X_validation = pca.transform(X_validation) # perform PCA on validation predictor vectors
    return X_train, X_validation


def fit_model(model, X_train, y_train):
    '''
    Fit the model to the data.
    '''
    model.fit(X_train, y_train) # fit model to training data
    return model


def evaluate_model(model, X_validation, y_validation):
    '''
    Evaluate the model's performance using root mean squared error and predictions on the validation set.
    '''
    y_predicted = model.predict(X_validation) # predict target on validation data
    rmse_validation = mean_squared_error(y_validation, y_predicted, squared=False) # calculate root mean squared error
    result = "{} RMSE = {}".format(model.__class__.__name__, round(rmse_validation, 4))
    return result

