#!/usr/bin/env python
# coding: utf-8

# In[17]:


#1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import numpy


# In[2]:


df=pd.read_csv(r"C:\Users\jgupt\Downloads\instagram_reach.csv",encoding = "ISO-8859-1")


# In[3]:


df= df.drop(['USERNAME', 'Caption'], axis=1)


# In[4]:


df1=df.copy()


# In[5]:


df1= df1.drop(['Hashtags'], axis=1)


# In[6]:


df1


# In[7]:


df1["Time since posted"]=df1["Time since posted"].str.replace("hours","")


# In[8]:


df1["Time since posted"]=df1["Time since posted"].astype(int)


# In[9]:


X = df1.iloc[:, :-2]


# In[10]:


y = df1.iloc[:, -2:]


# In[11]:


X=X.values


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[13]:


regressor1 = RandomForestRegressor()


# In[14]:


regressor1.fit(X_train,y_train)


# In[19]:


y_pred1 = regressor1.predict(X_test)


# In[20]:


r2 = r2_score(y_test, y_pred1)
print("R-squared Score:", r2)


# In[ ]:





# In[21]:


#2


# In[22]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import numpy


# In[23]:


df=pd.read_csv(r"C:\Users\jgupt\Downloads\archive (2)\ObesityDataSet_raw_and_data_sinthetic.csv")


# In[24]:


for i, (name, dtype) in enumerate(zip(df.columns, df.dtypes)):
    print(i, name, dtype)


# In[25]:


df = pd.get_dummies(df, columns=['Gender'])


# In[26]:


x=df.copy()
x


# In[27]:


x.drop("NObeyesdad", axis=1, inplace=True)


# In[28]:


y=df.copy()


# In[29]:


y= df["NObeyesdad"]


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[31]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


# In[32]:


cat_columns=[]
for i in X_train.columns:
    column_type = df[i].dtype
    if column_type=='object':
        print(i)
        column_obj=i
        cat_columns.append(i)


# In[33]:


num_type=[]
for i in X_train.columns:
    column_type = df[i].dtype
    if column_type!='object':
        column_obj=i
        num_type.append(i)
        
num_type


# In[34]:


cat_pipeline=Pipeline(

                steps=[
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
          )


# In[35]:


cat_pipeline=Pipeline(

                steps=[
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
          )


# In[37]:


num_pipeline= Pipeline(
              steps=[
              ("scaler",StandardScaler())

              ]
          )


# In[38]:


preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_type),
                ("cat_pipelines",cat_pipeline,cat_columns)

                ]
)


# In[39]:


X_train_preprocessed = preprocessor.fit_transform(X_train)


# In[40]:


X_test_preprocessed = preprocessor.transform(X_test)


# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# In[42]:


param_grid = {
    'n_estimators':100,  # Number of trees in the forest
    'max_depth': 10 , # Maximum depth of each tree
    'min_samples_split': 2,  # Minimum number of samples required to split an internal node
    'min_samples_leaf': 2  # Minimum number of samples required to be at a leaf node
}

# Create a Random Forest Classifier
rf_clf = RandomForestClassifier()

rf_clf = RandomForestClassifier(n_estimators=100,
                                max_depth=10,
                                min_samples_split=2,
                                min_samples_leaf=2)
rf_clf.fit(X_train_preprocessed, y_train)

# Make predictions on the test data
y_pred = rf_clf.predict(X_test_preprocessed)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[43]:


#3


# In[160]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


# In[161]:


df =pd.read_json("Downloads/News_Category_Dataset_v3.json",lines=True)


# In[162]:


df


# In[163]:


# Assuming you have the DataFrame 'df' with the required columns
df= df.drop(['link'], axis=1)


# In[164]:


df=df.drop(['short_description'], axis=1)


# In[165]:


df


# In[166]:


label_encoder = LabelEncoder()


# In[167]:


encoded_categories = label_encoder.fit_transform(df['category'])


# In[168]:


df['encoded_category'] = encoded_categories


# In[169]:


df


# In[170]:


df['processed_headline'] = df['headline'].str.lower().str.replace('[^\w\s]', '').str.split()
df['processed_authors'] = df['authors'].str.lower().str.replace('[^\w\s]', '').str.split()


# In[171]:


df


# In[172]:


df['text'] = df['processed_headline'].str.join(' ') + ' ' + df['processed_authors'].str.join(' ')


# In[173]:


df['text'][0]


# In[174]:


X = df.drop('encoded_category', axis=1)
y = df['encoded_category']


# In[175]:


X= X.drop(['headline'], axis=1)


# In[176]:


X


# In[178]:


y


# In[179]:


vectorizer = TfidfVectorizer()
X_ = vectorizer.fit_transform(X['text'])


# In[180]:


X_


# In[181]:


X_train, X_test, y_train, y_test = train_test_split(X_, df['encoded_category'], test_size=0.2, random_state=42)


# In[182]:


X_train


# In[183]:


knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_train)


# In[ ]:


svc = SVC()
svc.fit(X_train, y_train)


# In[ ]:


target_index = 0
target_data = X_test[target_index]


# In[ ]:


distances, indices = knn.kneighbors([target_data])
similar_data_knn = df.iloc[indices[0]]
print("Similar data using KNN:")
print(similar_data_knn)


# In[ ]:


decision_function = svc.decision_function([target_data])
similar_data_svc = df.iloc[decision_function.argsort()[::-1][:5]]
print("Similar data using SVC:")
print(similar_data_svc)


# In[1]:


#4


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


# In[2]:


df=pd.read_csv(r"C:\Users\jgupt\Downloads\archive (4)\online_shoppers_intention.csv")


# In[3]:


x=df.copy()
x


# In[4]:


x.drop("Informational_Duration", axis=1, inplace=True)


# In[5]:


x.drop("Revenue", axis=1, inplace=True)


# In[6]:


x.drop("Weekend", axis=1, inplace=True)


# In[7]:


y=df[["Informational_Duration","Revenue","Weekend"]]


# In[8]:


y_regression=y["Informational_Duration"]


# In[10]:


y_classification=y[["Revenue","Weekend"]]


# In[11]:


X_train, X_test, y_regression_train, y_regression_test, y_classification_train, y_classification_test = train_test_split(
    x, y_regression, y_classification, test_size=0.2, random_state=42
)


# In[12]:


X_train


# In[13]:


df["Revenue"].unique()


# In[14]:


cat_columnss=[]
for i in df.columns:
    column_type = df[i].dtype
    if column_type=='object':
        print(i)
        column_obj=i
        cat_columnss.append(i)


# In[15]:


num_type=["OperatingSystems","Browser","Region","TrafficType"]


# In[16]:


cat_pipeline=Pipeline(

                steps=[
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
          )


# In[17]:


num_pipeline= Pipeline(
              steps=[
              ("scaler",StandardScaler())

              ]
          )


# In[18]:


preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_type),
                ("cat_pipelines",cat_pipeline,cat_columnss)

                ]
)


# In[19]:


cat_y=["Revenue","Weekend"]


# In[20]:


cat_pipeline1=Pipeline(

                steps=[
                ("one_hot_encoder",OneHotEncoder())
                ]
          )


# In[21]:


preprocessor1=ColumnTransformer(
                [
                ("cat_pipelines",cat_pipeline,cat_columnss)

                ]
)


# In[22]:


X_train_preprocessed = preprocessor.fit_transform(X_train)


# In[23]:


X_test_preprocessed = preprocessor.transform(X_test)


# In[24]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[25]:


param_grid={
    "n_estimator":100,
    "n_jobs":1,
    "max_depth":10,
    "min_sample_split":2,
    "min_sample_leaf":2
}


# In[26]:


rf=RandomForestRegressor()


# In[27]:


rf = RandomForestRegressor(n_estimators=100,
                                max_depth=10,
                                min_samples_split=2,
                                min_samples_leaf=2)


# In[29]:


rf.fit(X_train_preprocessed,y_regression_train)


# In[30]:


X_train["VisitorType"].unique()


# In[31]:


y_pred = rf.predict(X_test_preprocessed)


# In[32]:


y_pred 


# In[34]:


classifier = RandomForestClassifier()
classifier.fit(X_train_preprocessed, y_classification_train)


# In[36]:


y_predicted_classification = classifier.predict(X_test_preprocessed)


# In[39]:


accuracy = accuracy_score(y_classification_test, y_predicted_classification)
print("Classification Accuracy:", accuracy)


# In[42]:


mse = mean_squared_error(y_regression_test, y_pred)
print("Mean Squared Error:", mse)


# In[43]:


r2 = r2_score(y_regression_test, y_pred)
print("R-squared:", r2)


# In[ ]:




