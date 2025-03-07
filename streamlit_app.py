import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    return pd.read_csv("train.csv")  # ✅ Correct usage inside a function

df = load_data()  # Call the function to get the cached data

st.title("Titanic : binary classification project")
st.sidebar.title("Table of contents")
pages=["Exploration", "DataVizualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0] : 
  st.write("### Presentation of data")
  st.dataframe(df.head(10))
  st.write(df.shape)
  st.dataframe(df.describe())

  if st.checkbox("Show NA") :
      st.dataframe(df.isna().sum())

if page == pages[1] : 
    st.write("### DataVizualization")

    fig = plt.figure()
    sns.countplot(x = 'Survived', data = df)
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x = 'Sex', data = df)
    plt.title("Distribution of the passengers gender")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x = 'Pclass', data = df)
    plt.title("Distribution of the passengers class")
    st.pyplot(fig)

    fig = sns.displot(x = 'Age', data = df)
    plt.title("Distribution of the passengers age")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x = 'Survived', hue='Sex', data = df)
    st.pyplot(fig)

    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig)

    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    st.pyplot(fig)

    df_numeric = df[['Survived', 'Pclass','Age', 'SibSp', 'Parch', 'Fare']]
    fig, ax = plt.subplots()
    sns.heatmap(df_numeric.corr(), ax=ax)
    st.pyplot(fig) 

if page == pages[2] : 
  st.write("### Modelling")

#remove the irrelevant variables
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

y = df['Survived']
X_cat = df[['Pclass', 'Sex',  'Embarked']]
X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

#replace the missing values for categorical variables by the mode and replace the missing values for numerical variables by the median.
for col in X_cat.columns:
  X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
for col in X_num.columns:
  X_num[col] = X_num[col].fillna(X_num[col].median())

#encode the categorical variables.(Ex: female/male ---> 0/1)
X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)

#concatenate the encoded explanatory variables without missing values to obtain a clean X dataframe
X = pd.concat([X_cat_scaled, X_num], axis = 1)

#separate the data into a train set and a test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#standardize the numerical values using the StandardScaler function (mean:0, Variance:1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

#create a function called prediction which takes the name of a classifier as an argument and which returns the trained classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def prediction(classifier):
    if classifier == 'Random Forest':
        clf = RandomForestClassifier()
    elif classifier == 'SVC':
        clf = SVC()
    elif classifier == 'Logistic Regression':
        clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf

#not unbalanced, it is interesting to look at the accuracy of the predictions
def scores(clf, choice):
    if choice == 'Accuracy':
        return clf.score(X_test, y_test)
    elif choice == 'Confusion matrix':
        return confusion_matrix(y_test, clf.predict(X_test))

#create a "select box" to choose which classifier to train
choice = ['Random Forest', 'SVC', 'Logistic Regression']
option = st.selectbox('Choice of the model', choice)
st.write('The chosen model is :', option)

clf = prediction(option)
display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
if display == 'Accuracy':
    st.write(scores(clf, display))
elif display == 'Confusion matrix':
    st.dataframe(scores(clf, display))


