# First of all lets import all the important libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics    import accuracy_score


# Application Heading
st.write(""" 
# Explore Different Machine Learning models and datasets
  Let's see which one performs better
""")

# Lets Create a Box in which we will be taking the name of the dataset which we will be using
dataset_name=st.sidebar.selectbox(
    "Select Dataset",
    ("Iris","Breast Cancer","Wine")

)


# Lets Create a Box in which we will be taking the name of the Machine learning Model which we will be using
classifier_name=st.sidebar.selectbox(
    "Select Classifier",
    ("KNN","SVM","Random Forest")

)

# Creating a function that will link up the actual dataset with the value selected from the list

def get_dataset(dataset_name):
    data=None
    if dataset_name=="Iris":
        data=datasets.load_iris()
    elif dataset_name=="Wine":
        data=datasets.load_wine()
    else:
        data=datasets.load_breast_cancer()
    x=data.data
    y=data.target
    return x,y

# Lets call this function here and then store the value of X and Y from the function output
X,y = get_dataset(dataset_name)


# Now we will print the shape of the dataset here on our application
st.write("Shape of the dataset",X.shape)
st.write("Number of classes",len(np.unique(y)))


def add_parameter_ui(classifier_name):
    params=dict() # Creates an empty dictionary
    if classifier_name=="SVM":
        C=st.sidebar.slider("C",0.01,10.0)
        params["C"]=C # It is degree of correct classification
    elif classifier_name=="KNN":
        K=st.sidebar.slider("K",1,15)
        params["K"]=K # It's the number of nearest neighbour
    else:
        max_depth=st.sidebar.slider("max_depth",2,15)
        params["max_depth"]=max_depth   # Depth of every tree grows in random forest
        n_estimators=st.sidebar.slider("n_estimator",1,100)
        params["n_estimators"]=n_estimators
    return params


# Now we will call this function and will make it equal to params variable
params=add_parameter_ui(classifier_name)


# Finally creating the function which will actually create a classifier for us and assign all the parameters being picked by user on runtime

def get_classifier(classifier_name,params):
    clf=None
    if classifier_name=="SVM":
        clf=SVC(C=params["C"])
    elif classifier_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])
    else:
        clf=RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=1234)
    return clf


# Now we will simply call this get_classifier method and store its value to a variable
clf=get_classifier(classifier_name,params)

# Now we are going to split our data into train/test with the ratio of 80/20
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,train_size=0.8)


# Now we will Train our classifier
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

# Checking Model Accuracy Score and then printing it out
acc=accuracy_score(y_test,y_pred)
st.write(f'Classifier={classifier_name}')
st.write(f'Accuracy={acc}')


# Plot Dataset
# We will be using PCA Technique to do the plotting because PCA allows us to reduce our features and show in a 2-dimensional way
# Basically PCA is feature reduction technique like doesn't matter how many features you have it helps you reduce it in a 2-dimensions.

pca=PCA(2)
X_projected=pca.fit_transform(X)

# Now we will slice our data in 0 or 1 dimension
x1=X_projected[:,0]
x2=X_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")

plt.xlabel("Principal Component 1 ")
plt.ylabel("Principal Component 2 ")
plt.colorbar()

#plt.show()
st.pyplot(fig)

