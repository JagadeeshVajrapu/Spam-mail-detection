import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

df= pd.read_csv('//enter your System path')
print(df)

data=df.where((pd.notnull(df)),'')
data.head()
data.head(10)
data.info()
data.shape

data.loc[data['Category']=='spam','Category',]=0
data.loc[data['Category']=='ham','Category',]=1
x=data['Message']
y=data['Category']
print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
print(x.shape)
print(x_train.shape)
print(x_test.shape)

print(y.shape)
print(y_train.shape)
print(y_test.shape)

feature_extraction = TfidfVectorizer()
x_train_features= feature_extraction.fit_transform(x_train)
x_test_features=feature_extraction.transform(x_test)

y_train=y_train.astype('int')
y_test=y_test.astype('int')
print(x_train)
print(x_train_features)

model=LogisticRegression()
model.fit(x_train_features,y_train)

prediction_on_training_data=model.predict(x_train_features)
accuracy_on_training_data=accuracy_score(y_train,prediction_on_training_data)

print("accuracy on training data is : ",accuracy_on_training_data)

prediction_on_test_data=model.predict(x_test_features)

accuracy_on_test_data=accuracy_score(y_test,prediction_on_test_data)
prediction_on_test_data=model.predict(x_test_features)

accuracy_on_test_data=accuracy_score(y_test,prediction_on_test_data)

input_yoour_email=["Enter email from your dataset"]
input_data_features=feature_extraction.transform(input_yoour_email)
prediction=model.predict(input_data_features)
print(prediction)
if(prediction==1):
  print('ham mail')
else:
  print('spam mail')









