import pandas as pd
from pprint import pprint
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
# profile = ProfileReport(df , title = 'Diabetes Report' , explorative = True)
# profile.to_file('diabetes_reprot.html')


target = "Outcome"
x = df.drop(target , axis = 1)
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 , random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #training SVM
# svm_model = SVC()
# svm_model.fit(x_train, y_train)
# svm_y_predict = svm_model.predict(x_test)
# print("SVM Model Accuracy: ", accuracy_score(y_test, svm_y_predict))
# print(classification_report(y_test, svm_y_predict))

# #training Logistic Regression
# logr_model = LogisticRegression()
# logr_model.fit(x_train, y_train)
# logr_y_predict = logr_model.predict(x_test)
# print("Logistic Regression Model Accuracy: ", accuracy_score(y_test, logr_y_predict))
# print(classification_report(y_test, logr_y_predict))

# #training Random Forest Classifier
# rf_model = RandomForestClassifier()
# rf_model.fit(x_train, y_train)
# rf_y_predict = rf_model.predict(x_test)
# print("Random Forest Model Accuracy: ", accuracy_score(y_test, rf_y_predict))
# print(classification_report(y_test, rf_y_predict))

# #grid search for hyperparameter tuning of Random Forest
# param_grid = {
#     'n_estimators': [100, 
#                      200, 
#                      300],
#     'criterion': ['gini', 
#                   'entropy' , 
#                   'log_loss']
# }

# grid_search = GridSearchCV(estimator=RandomForestClassifier() , 
#                            param_grid=param_grid , 
#                            cv=4 , 
#                            verbose=2 ,
#                            return_train_score=True ,
#                            scoring='accuracy')

# grid_search.fit(x_train , y_train)
# print(grid_search.best_estimator_)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

#LazyPredict
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)