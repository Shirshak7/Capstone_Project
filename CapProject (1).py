#Import Required Packages for EDA |
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/Users/shirshakbasnet/Programming/Capstone_Project_Customer/Customers.csv')

# # Sidebar options for analysis
analysis_options = st.sidebar.multiselect("Select the following for EDA:",
                                          options=["Gender", "Correlation", "Spending_Score", "Customer_by_Profession", "Family_Size"],
                                          default=["Gender", "Correlation", "Spending_Score", "Customer_by_Profession", "Family_Size"])

# Question1 - Counting Number of customers by gender(EDA)

num_plots = min(5, len(analysis_options))
for i in range(num_plots):
  if analysis_options[i] == "Gender":
    st.header("Gender Distribution")
    gender = df.Gender.value_counts()
    arr = np.array([gender[0],gender[1]])
    fig, ax = plt.subplots()
    plt.pie(arr,labels = ["Female","Male"],autopct = '%.1f%%',)
    st.pyplot(fig)
    st.write("The following pie chart shows the percentage of the customers by gender. It can be seen that the number of female customers is more than that of male.")


# #Question 2 (Correlation of customer between age and work experience)
  elif analysis_options[i] == "Correlation":
    age_exp = df[["Age", "Work_Experience"]].corr()

    #check correlation between variables
    st.header("Correlation between Age and Work Experience")
    sns.set(style="white") 
    fig,ax=plt.subplots()
    plt.rcParams['figure.figsize'] = (10, 5) 
    sns.heatmap(age_exp, annot = True, linewidths=.5, cmap="Blues")
    plt.title('Corelation Between Age and Work Experience', fontsize = 30)
    plt.show()
    st.pyplot(fig)
    st.write("The follwoing correlation figure shows the correlation between Age and Work Experience. From the figure it can be seen that this is a positive correlation which means as the age increases the work experience also increases.")

#Question 3 - Spending Score According to Gender
  elif analysis_options[i] == "Spending_Score":
    gen_spendsc = round(df.groupby("Gender")["Spending_Score_(1-100)"].mean(), 2)

    st.header("Spending Score of Customer by Gender")
    arr2 = np.array([gen_spendsc[0],gen_spendsc[1]])
    plt.xlabel('Spending_Score_(1-100)')
    plt.ylabel('Gender')
    fig, ax = plt.subplots()
    sns.barplot(x=["Male", "Female"], y= gen_spendsc)
    ax.set_ylabel("Spending_Score(1-100)")
    st.pyplot(fig)
    st.write("The folllowing bar graph shows the Spending Score(1-100) of customers by gender. It can be seen that the spending score of both male and female are pretty similar.")

#Question 4 - Customers by Profession
  elif analysis_options[i] == "Customer_by_Profession":
    st.header("Customers by Profession")
    # fig = plt.figure(figsize = (10,8))
    fig, ax = plt.subplots(figsize = (10, 8))
    g = sns.countplot(x = "Profession", data = df)
    g.set_ylabel('Customers')
    st.pyplot(fig)
    st.write("The following bar graph shows the distribution of customers by profession. It can be seen that the highest number of customer is from Artist profession followed by Healthcare, Entertainment,Engineer and so on.")


#Question 5 - Distribution of Family Size
  elif analysis_options[i] == "Family_Size":
    st.header("Distribution of Family Size")
    fig, ax = plt.subplots()
    family_sizes = df['Family_Size'].value_counts()
    sns.barplot(x=family_sizes.index, y=family_sizes.values)
    ax.set_xlabel('Family Size')
    ax.set_ylabel('Number of Customers')
    st.pyplot(fig)
    st.write("The following bar graph shows the family distribution of customers. It can be seen that the number of customers having family size is 2 is the highest and the number of customers having family size 8 is the lowest.")



#pre-processing
#encode object columns to integers
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder

for col in df:
  if df[col].dtype =='object':
    df[col]=OrdinalEncoder().fit_transform(df[col].values.reshape(-1,1))

class_label =df['Gender']
df = df.drop(['Gender'], axis =1)
df = (df-df.min())/(df.max()-df.min())
df['Gender']=class_label

#pre-processing
customer_data = df.copy()
le = preprocessing.LabelEncoder()
CustomerID = le.fit_transform(list(customer_data["CustomerID"]))
family_size = le.fit_transform(list(customer_data["Family_Size"])) 
age = le.fit_transform(list(customer_data["Age"])) 
annual_income = le.fit_transform(list(customer_data["Annual_Income"])) 
spending_score = le.fit_transform(list(customer_data["Spending_Score_(1-100)"])) 
profession = le.fit_transform(list(customer_data["Profession"])) 
work_experience = le.fit_transform(list(customer_data["Work_Experience"]))  
gender = le.fit_transform(list(customer_data["Gender"]))

# Predictive analytics model development by comparing different Scikit-learn classification algorithms
import sklearn

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


x = list(zip(CustomerID, family_size,age ,annual_income , spending_score, profession, work_experience))
y = list(gender)
# Test options and evaluation metric
num_folds = 5
seed = 7
scoring = 'accuracy'

# Model Test/Train
# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and vise versa for Y(features)
# The test data will test the accuracy of the model created
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
#splitting 20% of our data into test samples.

models = []
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
print("Performance on Training set")
for name, model in models:
  kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  msg += '\n'
  print(msg)


#Model Evaluation by testing with independent/external test data set. 
# Make predictions on validation/test dataset

svm = SVC()
gb = GradientBoostingClassifier()
rf = RandomForestClassifier()
nb = GaussianNB()

best_model = svm

best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
print("Best Model Accuracy Score on Test Set:", accuracy_score(y_test, y_pred))

#Model Performance Evaluation Metric 1 - Classification Report
print(classification_report(y_test, y_pred))

#Confusion matrix 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



#Model Evaluation Metric 4-prediction report
for x in range(len(y_pred)):
  print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x],)


st.subheader("The following part is for PDA.")


# Algorithm Comparison
st.header("Algorithm Comparision")
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
st.pyplot(fig)
st.write('''For the training part I'have used 4 different models: GaussianNB, SVC, GradientBoostingClassifier,
            and RandomFOrestClassifier. From the figure it can be seen that SVM is the best model with lowest variance.
            The SVM model had the highest accuracy score of 0.59''')


# Confusion Matrix
st.header("Confusion Matrix")
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
st.pyplot(fig)
st.write("The following figure is a confusion matrix. It shows that for the 232 Female it is predicting correctly and for the 168 males it is predicting correctly. ")

