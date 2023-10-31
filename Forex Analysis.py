import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style



df = pd.read_csv("forex.csv", index_col='date')
df.head()
df.index = pd.to_datetime(df.index)

#Get Number Of All The Records concerning the USD to  EGP Exchange
print('Data has '+str(len(df[df.slug=='USD/EGP']))+' observations per exchange rate')
sns.set()
style.use('seaborn-darkgrid')


def currency_plotter(slug):
    plt.figure(figsize=(15, 8))
    plt.plot(df[df.slug==slug].close,'g',  linewidth=0.8,label=slug+' Close')
    plt.title(slug)
    plt.legend()
    plt.show()
    
def country_plotter(country_coin_name, country_name):
    plt.figure(figsize=(15, 8))        
    plt.title(country_name)
    country_df = df[df.currency==country_coin_name]
    for slug in country_df.slug.unique():
        if country_df[country_df.slug==slug].close.mean() > 0.5:
            plt.plot(country_df[country_df.slug==slug].close, linewidth=0.8, label=slug)
        else:
            plt.plot(country_df[country_df.slug==slug].close*100, linewidth=0.8, label=(slug+' *10^2'))

    plt.legend()
    plt.show()
    return country_df

#Plot The Exchange Rate For USD To EGP & The Egyptian currency Rate over the  years 
currency_plotter('USD/EGP')
country_plotter('EGP', 'Egypt')



print(df.corr())
sns.heatmap(df.corr())
plt.show()

#Get Only The Data Concerning THE USD to EGP Exchange 
#and getting the realtive information to the prediction of the closing value
for x in df['slug']:
    if(x=='USD/EGP'):
        print(df['currency'])
        X = df[["open", "high", "low"]]
        Y = df["close"]
        
X = X.to_numpy()
Y = Y.to_numpy()
Y = Y.reshape(-1, 1)

#Split the data to train and test given 70% for the train
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=42)


#Prediction Using the DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

df = pd.DataFrame(data={"Predicted Rate": ypred.flatten()})
print(df.head())
print(mean_squared_error(ytest,ypred))
plt.plot(ytest,color ='k')
plt.plot(ypred,color ='r')
plt.show()



#Prediction Using the LinearRegression
from sklearn.linear_model import LinearRegression

regr = LinearRegression()
  
regr.fit(xtrain, ytrain)

print(regr.score(xtest, ytest))


y_pred = regr.predict(xtest)
print(mean_squared_error(ytest,y_pred))
plt.plot(ytest,color ='k')
plt.plot(y_pred,color ='r')
plt.show()

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(xtrain)
X_test = sc.transform(xtest)

#Prediction Using the RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(n_estimators=100,max_depth=3)
regr.fit(X_train, ytrain)

y_predz = regr.predict(xtest)

print(mean_squared_error(ytest,y_predz))
plt.plot(ytest,color ='k')
plt.plot(ypred,color ='r')
plt.show()


#Prediction Using the Support Vector
from sklearn.svm import SVC

svclassifier = SVC(kernel='rbf', random_state=0, gamma=0.001, C=10)

svclassifier.fit(X_train, ytrain)


y_predax = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(ytest,y_predax))
print(classification_report(ytest,y_predax))
print(accuracy_score(ytest, y_predax))



#Prediction Using the Support Vector Regression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

svr = SVR()

clf = GridSearchCV(svr, parameters,cv=5)

clf.fit(xtrain, ytrain)


print(clf.score(xtest, ytest))


y_predx = clf.predict(xtest)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(ytest,y_predx))
plt.plot(ytest,color ='k')
plt.plot(y_predx,color ='r')
plt.show()


#Prediction Using the Logistic Regression
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(C=0.7, penalty='l2', tol=0.0001, solver='saga')
    
logisticRegr.fit(X_train, ytrain)


y_presd = logisticRegr.predict(xtest)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(ytest,y_presd))
print(classification_report(ytest,y_presd))
print(accuracy_score(ytest, y_presd))