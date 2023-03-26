import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wine=pd.read_csv('https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv')
wine

## preprocessing

# convert data into target and features
y=wine['Wine']
y # target

X=wine.drop(['Wine'],axis=1)
X # features

X.shape, y.shape

## step 1 : standardization


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled

## step 2: construction of covariance matrix

cm=np.cov(X_scaled.T) # convert observation into data points, m has to be a variable so we transpose
cm

cm.shape

X_scaled.shape

## step 3: finding eigen value, eigen vector

eig_val,eig_vec=np.linalg.eig(cm)
eig_val # 13 values

eig_vec

eig_vec.shape

eig_val.shape

## step 4 : sorting eigen values

sorted_eig_val=[i for i in sorted(eig_val,reverse=True)]
sorted_eig_val

## step 5: identify the no of PCA according to the error value / significance value

#### for simplicity we choose no of PCA as 2 for the sake of visualization

tot=sum(sorted_eig_val)
tot

exp_var=[(i/tot) for i in sorted_eig_val ] # gives weightage of each of the eigen value by taking out the percentages
exp_var 

cum_exp_var=np.cumsum(exp_var)
cum_exp_var

# last 3 can be ignored as threshold is 95%

## step 6:  Plotting

plt.bar(range(1,14),exp_var,label='Explained Variance')
plt.xlabel('Principle component')
plt.ylabel('Explained variance')
# plt.show()
plt.legend()

## step 7: construction of projection matrix

# for now our focus is only on first 2. only 55% can be explained

# create a pair of vector and value
eigen_pair=[(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(len(eig_val))]
eigen_pair

'''
[(4.732436977583593,
  array([-0.1443294 ,  0.24518758,  0.00205106,  0.23932041, -0.14199204,
         -0.39466085, -0.4229343 ,  0.2985331 , -0.31342949,  0.0886167 ,
         -0.29671456, -0.37616741, -0.28675223])),
 (2.511080929645125,
  array([2.51108093, 2.51108093, 2.51108093, 2.51108093, 2.51108093,
         2.51108093, 2.51108093, 2.51108093, 2.51108093, 2.51108093,
         2.51108093, 2.51108093, 2.51108093])),
'''

eigen_pair=[(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(0,13)]
eigen_pair

# eigen_pair 0th col shows eigen value
# hstack is horizontal stack

w=np.hstack((eigen_pair[0][1][:,np.newaxis],eigen_pair[1][1][:,np.newaxis])) 
# 0th row, 1st column ie array, : shows all the values in the array
# we take 0 and 1 as we consider first two records

w

w.shape

## step 8: transforming 13D into 2D

X_scaled.shape, w.shape

# 178 observations explained using 13 features reduced to 2 

new_x=X_scaled.dot(w)
new_x.shape

## step 9: visualizing the projected data 

for l in np.unique(y):
#     plt.scatter(new_x[y==1,0],new_x[y==1,0],marker='s')
#     plt.scatter(new_x[y==2,0],new_x[y==2,0],marker='x')
    plt.scatter(new_x[y==3,0],new_x[y==3,0],marker='o')
    

## using sklearn

from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
X_pca=pca.fit_transform(X_scaled)

pca.components_.T[:,1]

pca.explained_variance_ratio_
-------------------------===================================----------------------------==============================
session 2
## locally linear embedding LLE

# used when projection fails eg swiss roll, find other reasons

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_swiss_roll
X,y=make_swiss_roll(n_samples=1000,random_state=100)
X

y

plt.scatter(X[:,0],X[:,1],c=y)

## building the model

from sklearn.manifold import LocallyLinearEmbedding
lle=LocallyLinearEmbedding(n_neighbors=5,random_state=100)

lle

## transforming the data

X_lle=lle.fit_transform(X)
X_lle

plt.scatter(X_lle[:,0],X_lle[:,1],c=y)

lle=LocallyLinearEmbedding(n_neighbors=10,random_state=100)

X_lle=lle.fit_transform(X)
X_lle
plt.scatter(X_lle[:,0],X_lle[:,1],c=y)

## exercise : perform LLE on wine dataset

-------------------------========================-------------------------===========================--------------------------
session 3
# TSNE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml    #  fetch openml class provides many high dimensional data

## Accessing the dataset

X,y=fetch_openml('mnist_784',version=1,return_X_y=True)

X.shape

y.shape

X.head() #initials are zero coz lots of free spaces

X.iloc[1]

y.value_counts()

## Plotting the images

plt.imshow(X.iloc[1]) # have to convert pandas X to numpy as reshape is in numpy so will throw an error

plt.imshow(X.iloc[1].to_numpy().reshape(28,28))
plt.title(y[1]);

plt.imshow(X.iloc[10].to_numpy().reshape(28,28))
plt.title(y[10]);

plt.imshow(X.iloc[100].to_numpy().reshape(28,28))
plt.title(y[100]);    #unclear 5 . feature provided is not properly distinguished in all the cases uptill now

plt.imshow(X.iloc[500].to_numpy().reshape(28,28))
plt.title(y[500]);

plt.imshow(X.iloc[1500].to_numpy().reshape(28,28))
plt.title(y[1500]);

plt.imshow(X.iloc[15000].to_numpy().reshape(28,28))
plt.title(y[15000]);

plt.imshow(X.iloc[25000].to_numpy().reshape(28,28))
plt.title(y[25000]);

plt.imshow(X.iloc[45000].to_numpy().reshape(28,28))
plt.title(y[45000]);

## Creating a random sample of 1k from 70k

np.random.seed(100)
sample=np.random.choice(X.shape[0],1000) # we want 1000 random samples
print(sample)

## Creating a subset of 1k samples of X,y

X1=X.iloc[sample,:]
X1.shape

X1.head() # respective datapoints are chosen .. correesponding tp 38408... so on..

y1=y[sample]
y1.shape

y1.head()

## Building TSNE model

from sklearn.manifold import TSNE

tsne=TSNE(n_components=2,perplexity=30)  # 2 is the dimension to which we want to reduce the big dmensional dataset

X_tsne=tsne.fit_transform(X1)
X_tsne.shape

X_tsne[0]  # 2d value earlier which was of 784d

## Plotting transformed data points

plt.scatter(X_tsne[:,0],X_tsne[:,1]);  # no sense - we wanted to have digits

plt.scatter(X_tsne[:,0],X_tsne[:,1],c=y1.astype(float)); # again not clear So have to convert it into a datafarme

## Creating a DF

X_df=pd.DataFrame({'X0':X_tsne[:,0],
                  'X1':X_tsne[:,1],
                  'Label':y1})
X_df                                          # corrresponding to label we get diff color

plt.figure(figsize=(15,12))
sns.lmplot(data=X_df,x='X0',y='X1',hue='Label')  # lines are coming

plt.figure(figsize=(15,12))
sns.lmplot(data=X_df,x='X0',y='X1',hue='Label',fit_reg=False)

## Exercise

Perform TSNE on wine dataset.

# Its small so need to take the sample from that. You can do it directly

-------------------========================---------------------------------------------===================================
session 4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## accessing the dataset

cust=pd.read_csv('wholesale_customers.csv')

cust

cust.shape # 440 customers 8 features

## standardization

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x=sc.fit_transform(cust)
x

## convert array to Dataframe


X=pd.DataFrame(x,columns=cust.columns)
X

## draw dendrogram

# we cannot draw dendrogram using sklearn
# use scipy tp draw dendrogram
import scipy.cluster.hierarchy as sch

dendro=sch.dendrogram(sch.linkage(X,method='ward')) # ward for max
dendro

plt.figure(figsize=(15,12))
dendro=sch.dendrogram(sch.linkage(X,method='ward'))
plt.axhline(y=35,color='red',linestyle='--') # draw line parallel to any axis

plt.figure(figsize=(15,12))
dendro=sch.dendrogram(sch.linkage(X,method='ward'))
plt.axhline(y=12,color='red',linestyle='--')
plt.axhline(y=10,color='blue',linestyle='-.')

# clusters cover how much percent of the total samples
# as a marketing professional

# Agglomerative clustering using sklearn

from sklearn.cluster import AgglomerativeClustering

clust=AgglomerativeClustering(n_clusters=2,linkage='ward')

clust.fit_predict(X)

clust.labels_

## adding cluster labels to the dataframes

X['label']=clust.labels_
X

# analysis of the segmentation

# find the no of customers in each segmentation
X.value_counts()

X.value_counts

X.value_counts('label')

X['label'].value_counts()

# list the samples belonging to label 0
X[X['label']==0]

# list the samples belonging to label 1
X[X['label']==1]

# find the buying pattern of milk and grocery wrt to segments
plt.scatter(X['Grocery'],X['Milk'])

sns.scatterplot(y=X['Grocery'],x=X['Milk'],hue=X['label'])

#  if we draw a line at y=2, that means who people who buy grocery below 2, all belong to label 1

sns.pairplot(data=X,hue='label')

# channel - mixture of gaussian model
# delicassen - mis of gaussion graphs
# fresh vs frozen - most of the people belong to one class
=----------------------------------------====================================================------------------
session 6
## mall customer segmentation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## access dataset

customer=pd.read_csv('Mall_customers.csv')

customer

# unsupervised approach

# ignore customer id


## preprocessing

customer=customer.drop(['CustomerID'],axis=1)
customer

## one hot encoding for gender : convert categorical to numeric

cust=pd.get_dummies(customer)
cust

# standardization

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(cust)
X

# convert into dataframe

X=pd.DataFrame(X,columns=cust.columns)

X

# Dimensionality reduction using tsne

from sklearn.manifold import TSNE
tsne=TSNE(n_components=2,random_state=100) # feature=2
type(tsne)

X_tsne=tsne.fit_transform(X) #converted 5D data into 2D using tsne model
X_tsne

plt.scatter(X_tsne[:,0],X_tsne[:,1],c=X_tsne[:,1])

 # properly defined 2 clusters are formed
    # is tsne is good, that means if it retains the characteristics

## findings

1. formation of clearly defined clusters.
2. there can be 2 clusters

## clustering using GMM

from sklearn.mixture import GaussianMixture
n_comps=np.arange(1,20,1)
aic_score=[]
bic_score=[]
for n in n_comps:
    model=GaussianMixture(n_components=n,random_state=10,n_init=5)
    model.fit(X)
    aic_score.append(model.aic(X))
    bic_score.append(model.bic(X))    

print(aic_score)

print(bic_score)

plt.plot(n_comps,aic_score,c='b',label='AIC')
plt.plot(n_comps,bic_score,c='g',label='BIC')
plt.legend();

## findings
1. no of clusters=2

##  building GM model with 2 clusters

gm=GaussianMixture(n_components=2,random_state=100,n_init=5)
gm.fit(X)

pred=gm.predict(X)
pred

gm.means_ # means of the distributions

gm.covariances_

gm.weights_ # percentage of class labels

## adding label columns to dataframe

customer['label']=pred
customer

customer[customer['label']==1]

## insights

customer['label'].value_counts

customer['label'].value_counts()

customer['label'].value_counts()/sum(customer['label'].value_counts())

customer[customer['label']==0]

customer[customer['Gender']=='Male']

customer[customer['label']==0][customer['Gender']=='Male']

1. all males form a segment with 44% weightage of customer base
2. all females form a segment with 56% weightage of customer base


## forming samples
gm.sample(1000)

# generative model
---------------------------------===========================================================-----------------------------
session 5

