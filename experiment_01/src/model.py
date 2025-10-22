from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def train_random_forest(x_train, y_train, n_estimators=100, random_state=42):
    """训练随机森林回归模型"""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(x_train, y_train)
    return model

# 其它模型
def train_linear_regression(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def train_svr(x_train, y_train):
    model = SVR()
    model.fit(x_train, y_train)
    return model




