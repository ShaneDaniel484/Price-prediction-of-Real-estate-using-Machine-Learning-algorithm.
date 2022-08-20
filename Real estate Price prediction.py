#!/usr/bin/env python
# coding: utf-8

# # <span style="color:black">Price prediction of Real estate using Machine Learning algorithm. 
# 
# 
# 
# ![image-3.png](attachment:image-3.png)
# 
# 
# 
# 
# </span> 

# ## BY <span style="color:purple">Shane.D</span> Batch:2021-6832

# ## <span style="color:blue">Problem Statement</span>

# We have to predict the <span style="color:red">price of the real estate properties properties based on their total square feet area , number of bathrooms and number of bedrooms.</span>

# ## <span style="color:blue">Dataset description</span>

# This dataset has been taken from <span style="color:red">kaggle.com</span>

# This dataset contains <span style="color:red">13321</span> rows and <span style="color:red">9</span> columns

# The following are the  of <span style="color:red">detailed description</span> the variables 

# 1.<span style="color:green">area_type</span>- Describes about the saleable area of the property

# 2.<span style="color:green">availability</span>- Tells when the property is ready to move in

# 3.<span style="color:green">location</span>- Tells the location of our property

# 4.<span style="color:green">size</span>- Tells the number of bhk 

# 5.<span style="color:green">society</span>- Tells what type of society the property is in

# 6.<span style="color:green">total_sqft</span>- Gives the total square feet area

# 7.<span style="color:green">bath</span>-Tells the number of bathrooms

# 8.<span style="color:green">balcony</span>-Tells the number of balcony

# 9.<span style="color:green">price</span>-This is our <span style="color:red">target variable</span> which we have to predict.
# We have to take our price as lakhs in Indian rupees

# ## <span style="color:blue">Importing necessary libraries</span>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ##  <span style="color:blue">Data loading</span>

#  <span style="color:red">importing the file</span>

# In[2]:


df=pd.read_csv("C:\\Users\\aaron\\Downloads\\Real estate price prediction\\real estate.csv")
df.head()


#  <span style="color:red">Checking the shape of our dataset</span>

# In[3]:


df.shape


#  <span style="color:red">Grouping the area type and getting the count of each area type</span>

# In[4]:


df.groupby('area_type')['area_type'].agg('count')


#  <span style="color:red">Dropping features that are not required to build our model</span>

# In[5]:


df=df.drop(['area_type','society','balcony','availability'],axis='columns')
df.head() 


# ## <span style="color:blue"> Data cleaning</span>

#  <span style="color:red">Checking the null values</span>

# In[6]:


df.isnull().sum()


#  <span style="color:red">we are dropping the null values</span>

# In[7]:


df=df.dropna()
df.isnull().sum()


# ## <span style="color:blue"> Feature engineering</span>

#  <span style="color:red">Creating additional columns from existing colums to make it less complicated</span>

#  <span style="color:red">Checking the unique values in size column</span>

# In[8]:


df['size'].unique()


# <span style="color:red"> we noticed that the values in size column are incosistent
#  so we are creating a new column "bhk" with int datatype of size column </span>

#  <span style="color:red">Here we create a lambda function to split the size column and take the first value</span>

# In[9]:


df['bhk']=df['size'].apply(lambda x: int(x.split(" ")[0]))
df.head()


# <span style="color:red">Checking the unique values in bhk column </span>

# In[10]:


df['bhk'].unique()


#  <span style="color:red">Checking the unique values in total_sqft column</span>

# In[11]:


df.total_sqft.unique()


#  <span style="color:red">Here we noticed some values are given as a range.
# so we create a function to extract only the float values</span>

# In[12]:


def isfloat(x):
    try:
        float(x) #to convert the values to float
    except:
        return False #if its not a valid number it will return false
    return True #else it will return true


#  <span style="color:red">Here we are using ~ "negate operator" to get the values which are false in the data</span>

# In[13]:


df[~df['total_sqft'].apply(isfloat)].head(10)


#  <span style="color:red">Here we create a function to convert those ranges to its mean</span>

# In[14]:


def convt(x):
    spt=x.split(" - ")# we are splitting the numbers at -
    if len(spt) ==2: #to get only the ranges and exclude the string values
        return(float(spt[0])+float(spt[1]))/2 #to find the mean
    try:
        return float(x) #if its not a range return a float value
    except:
        return None #if its not a valid value return nothing


#  <span style="color:red">Testing the function we created</span>

# In[15]:


convt('2354') #converts the number to float


# In[16]:


convt('500 - 1000') #converts the range to its mean


# In[17]:


convt('34.46Sq. Meter') #returns nothing if the value is not valid


#  <span style="color:red">We are applying the function we have created to the dataset</span>

# In[18]:


df['total_sqft']=df['total_sqft'].apply(convt)
df.head()


#  <span style="color:red">Now we have modified our total_sqft</span>

# In[19]:


df.total_sqft.unique()


#  <span style="color:red">We are cross verifying the output of our function</span>

# In[20]:


df.loc[30]


#  <span style="color:red">In the index no.30 we had the sqft value as 2100-2850</span>

# In[21]:


(2100+2850)/2 #checking the mean value for verifying


#  <span style="color:red">We are creating a new column "Price_per_sqft" with price and total_sqft column</span>

# In[22]:


df['price_per_sqft']=df['price']*100000/df['total_sqft']
df.head()


# ## <span style="color:blue"> Exploratory Data Analysis</span>

#   <span style="color:red">Since we have a lot of values it is complicated to perform eda,so we are filtering some data</span>

# In[23]:


df.location.unique()


# In[24]:


len(df.location)


#   <span style="color:red">We will filter the top three locations and perform eda</span>

# In[25]:


df.location.value_counts()


#   <span style="color:red">We will store it in a new dataframe</span>

# In[26]:


dfw=df[df.location=='Whitefield']
dfs=df[df.location=='Sarjapur  Road']
dfe=df[df.location=='Electronic City']


# In[27]:


print("The shape dfw dataframe is:",dfw.shape)
print("The shape dfs dataframe is:",dfs.shape)
print("The shape dfe dataframe is:",dfe.shape)


#  <span style="color:red">Now we will plot a line plot between price and sqft with those three datasets we have filtered</span>

# In[28]:


plt.figure(figsize=(10,10))
sns.lineplot(x="total_sqft",y="price",data=dfw)
plt.title("PRICE VS SQFT")
plt.show


# In[29]:


plt.figure(figsize=(10,10))
sns.lineplot(x="total_sqft",y="price",data=dfs)
plt.title("PRICE VS SQFT")
plt.show


# In[30]:


plt.figure(figsize=(10,10))
sns.lineplot(x="total_sqft",y="price",data=dfe)
plt.title("PRICE VS SQFT")
plt.show


# From the above graph we can infer that the price increase with increase in total_sqft but there are some value which have a lower value despit having a larger sqft area

# Now we will plot a line plot between price and bath

# In[31]:


plt.figure(figsize=(10,10))
sns.lineplot(x="bath",y="price",data=dfw)
plt.title("PRICE VS BATH")
plt.show


# In[32]:


plt.figure(figsize=(10,10))
sns.lineplot(x="bath",y="price",data=dfs)
plt.title("PRICE VS BATH")
plt.show


# In[33]:


plt.figure(figsize=(10,10))
sns.lineplot(x="bath",y="price",data=dfe)
plt.title("PRICE VS BATH")
plt.show


# Now we will plot a line plot between price and bhk

# In[34]:


plt.figure(figsize=(10,10))
sns.lineplot(x="bhk",y="price",data=dfw)
plt.title("PRICE VS BHK")
plt.show


# In[35]:


plt.figure(figsize=(10,10))
sns.lineplot(x="bhk",y="price",data=dfs)
plt.title("PRICE VS BHK")
plt.show


# In[36]:


plt.figure(figsize=(10,10))
sns.lineplot(x="bhk",y="price",data=dfe)
plt.title("PRICE VS BHK")
plt.show


# from the above six graphs we can clearly see that there is a constant increase in price with increase in number of bedrooms and bathrooms

# Here we are plotting a histogram for the number of bedrooms in each location

# In[37]:


plt.figure(figsize=(10,10))
plt.hist(dfw.bhk)
plt.title("No. of bedrooms")
plt.show()


# In[38]:


plt.figure(figsize=(10,10))
plt.hist(dfs.bhk)
plt.title("No. of bedrooms")
plt.show()


# In[39]:


plt.figure(figsize=(10,10))
plt.hist(dfe.bhk)
plt.title("No. of bedrooms")
plt.show()


# Here we can see that the price is correlated with the total_sqft,bath and bhk.
# We are not considering the price_per_sqft column because it is derived from total_sqft and price column.

# In[40]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
plt.title("Heatmap")
plt.show()


# ##  <span style="color:blue"> Dimensionality reduction</span>

#  <span style="color:red">Here we are reducing the number of random variables in a problem by obtaining a set of principal variables</span>

#  <span style="color:red">Checking the unique values in location</span>

# In[41]:


df.location.unique()


# <span style="color:red">There are many locations so we are using len function to get the length </span>

# In[42]:


len(df.location.unique())


#  <span style="color:red">We are using the strip function in location column to clear the extra spaces</span>

# In[43]:


df.location=df.location.apply(lambda x:x.strip())


#  <span style="color:red">We are grouping the locations and getting the count of each location in decending order</span>

# In[44]:


locnum=df.groupby('location')['location'].agg('count').sort_values(ascending= False)
locnum


# In[45]:


locnum.values.sum() #there are a total of 13246 values


#  <span style="color:red">Since there are 13246 locations we are going to group the location which have count less than 10 to other</span>

# In[46]:


locnum_less_than_10=locnum[locnum<=10]


# In[47]:


locnum_less_than_10


# In[48]:


len(df.location.unique())


#  <span style="color:red"><span style="color:red"></span></span>

# In[49]:


df.location=df.location.apply(lambda x: 'other' if x in locnum_less_than_10 else x)


#  <span style="color:red">Now we have reduced the number of unique values from 1293 to 242</span>

# In[50]:


len(df.location.unique())


#  <span style="color:red">Here we can notice that in 9th row we got other</span>

# In[51]:


df.head(10)


# ##  <span style="color:blue">Outlier removal</span>

#  <span style="color:red">First we will remove the outliers by basic logic and then we will use standard deviation to remove the outliers</span>

#  <span style="color:red">We are applying a basic logic here that is a bedroom should have atleast 300 sqft</span>

# In[52]:


df[df.total_sqft/df.bhk<300].head()


# <span style="color:red">We have 6 bhk apartment with 1020 sqft. Another one is 8 bhk and total sqft is 600. These are data errors and can be removed </span>

# In[53]:


df=df[~(df.total_sqft/df.bhk<300)] #we are dropping those outliers


# In[54]:


df.shape #after dropping the outliers by logic we are left with 12502 rows


#  <span style="color:red">Here we are removing outliers with standard deviation and mean</span>

#  <span style="color:red">First we will get a five point summary using describe function</span>

# In[55]:


df.price_per_sqft.describe()


#  <span style="color:red">We are creating a function to remove the outliers</span>

# In[56]:


def rem_out(df):#here we are passing our dataframe
    df_out = pd.DataFrame()#create a output dataframe
    for key, subdf in df.groupby('location'):#key and the sub dataframe in our df grouped by location
        m = np.mean(subdf.price_per_sqft)#mean function
        sd = np.std(subdf.price_per_sqft)#standard deviation function
        #create a variable with values greater than difference of mean and sd with values less than sum of mean and sd
        reduced_df = subdf[(subdf.price_per_sqft>(m-sd)) & (subdf.price_per_sqft<=(m+sd))]
        #now we concatenate the dfs into df_out
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out


#   <span style="color:red">Passing our dataframe in this function</span>

# In[57]:


df=rem_out(df)


# In[58]:


df.shape #after dropping the outliers by Std. deviation & mean we are left with 10241 rows


#  <span style="color:red">Here we are creating a function to plot the scatter plot</span>

# In[59]:


def plot_scatter_chart(df,location):#passing the dataframe and location
    bhk2 = df[(df.location==location) & (df.bhk==2)]#for 2bhk
    bhk3 = df[(df.location==location) & (df.bhk==3)]#for 3bhk
    matplotlib.rcParams['figure.figsize'] = (15,10)#assigning the figsize
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (in Lakhs)")
    plt.title(location)
    plt.legend()


#  <span style="color:red">We can notice that the price of 3bhk is less than 2bhk</span>

# In[60]:


plot_scatter_chart(df,'Rajaji Nagar')


# In[61]:


plot_scatter_chart(df,'Hebbal')


#  <span style="color:red">Now we create a function to remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment</span>

# In[62]:


def remove_bhk_outliers(df):#passing our dataframe
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):#we are grouping the df by location
        bhk_stats = {} 
        for bhk, bhk_df in location_df.groupby('bhk'):#after we group the df by location we group it again by bhk
            #we will build a dictionary
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        #here we are checking whether the price of say 3 bhk is less than 2 bhk,so we create a loop to check all bhks    
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)#this will help to check bhk less than the passed value
            if stats and stats['count']>5:
                #we are ecluding the values whose price is less than the mean of the previous bhk
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')


#  <span style="color:red">We are passing the function in out dataset</span>

# In[63]:


df=remove_bhk_outliers(df)


# In[64]:


df.shape #we have reduced the outliers and we have reduced our df shape from 10241 to 7329


#  <span style="color:red">Here we can see that the values are cleared,but still some values are present
# but those are hard to remove so we are neglecting that</span>

# In[65]:


plot_scatter_chart(df,'Hebbal')


# <span style="color:red">We can see a normal distribution of our data between 0 to 10000.When we plot a histogram for the count of price per square feet </span>

# In[66]:


plt.hist(df.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


#  <span style="color:red">We will check the unique values in bath column</span>

# In[67]:


df.bath.unique()


#  <span style="color:red">We are checking the number of bathrooms greater than 10</span>

# In[68]:


df[df.bath>10]


#  <span style="color:red">In the above table we can see that 16 bathrooms for 10000sqft and 12 bathrooms for 4000 sqft which is clearly an error,
# also consider the number of bedrooms</span>

#  <span style="color:red">Here we are plotting a histogram for the count of bathroom</span>

# In[69]:


plt.hist(df.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


#  <span style="color:red">Here we are considering the no of bathrooms 2 greater than no of bedrooms as outliers</span>

# In[70]:


df[df.bath>df.bhk+2]


#  <span style="color:red">We are dropping those values with more number of bathrooms</span>

# In[71]:


df2=df[df.bath<df.bhk+2]


# In[72]:


df2.shape #we have reduced the outliers and we have reduced our df shape from 7329 to 7251


# In[73]:


df=df2 #keeping everydataset in same name


#  <span style="color:red">Dropping some features which are not necessary</span>

# In[74]:


df=df.drop(['size','price_per_sqft'],axis='columns')
df.head()


#  <span style="color:red">We are creating a column for each of the location using pd.dummies</span>

# In[75]:


dummies=pd.get_dummies(df.location)
dummies.head()


#  <span style="color:red">We are concatenating the df and dummies,also we are dropping the "other" column here</span>

# In[76]:


df=pd.concat([df,dummies.drop('other',axis='columns')],axis='columns')
df.head()


#  <span style="color:red">We will also drop the location column since we have created columns for each location seperately</span>

# In[77]:


df=df.drop('location',axis='columns')
df.head()


# ##  <span style="color:blue">Building our model</span>

#  <span style="color:red">Assigning the independent variables to x</span>

# In[78]:


x=df.drop('price',axis='columns')
x.head()


#  <span style="color:red">Assigning the dependent variables to y</span>

# In[79]:


y=df.price
y.head()


#  <span style="color:red">Importing train test split to split our dataset</span>

# In[80]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)


# <span style="color:red">We are creating a linear regression model</span>

# In[81]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
prediction=lr.predict(x_test)
print('The accuracy is: ',lr.score(x_test,y_test)*100)


# In[82]:


plt.figure(figsize=(10,10))
sns.distplot(y_test,color= 'r')
sns.distplot(prediction)
plt.show()


# Here we can see that the <span style="color:red">red color represents true values</span> and  <span style="color:blue">blue color represents predicted value</span>

#  <span style="color:red">Here we use K Fold cross validation to measure accuracy of our LinearRegression model</span>

# In[83]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), x, y, cv=cv)


#  <span style="color:red">We are getting an accuracy of over 80% but we can also check other models</span>

#  <span style="color:red">To find best model we are using GridSearchCV</span>

# In[84]:


from sklearn.model_selection import GridSearchCV #importing gridsearch 
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(x,y):#we create a function to use gridsearch
#here we pass different models with different hyperparameters
    alg = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for alg_name, config in alg.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': alg_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(x,y)


#  <span style="color:red">Based on the result we can say the Linear regression has best scores,so we will use that</span>

#  <span style="color:red">Here we create a function to predict the price</span>

# In[85]:


def predict_price(location,sqft,bath,bhk):#we are passing these values  
    loc_index = np.where(x.columns==location)[0][0] #gets the index of the particular location

    z = np.zeros(len(x.columns))
    z[0] = sqft
    z[1] = bath
    z[2] = bhk
    if loc_index >= 0:
        z[loc_index] = 1

    return lr.predict([z])[0]


# In[86]:


x.columns #column in x


#  <span style="color:red">Now we will predict some prices</span>

# In[87]:


predict_price('1st Phase JP Nagar',1000,2,2)


# In[88]:


predict_price('1st Phase JP Nagar',1000,3,3)


# In[89]:


predict_price('Indira Nagar',1000,2,2)


# In[90]:


predict_price('Indira Nagar',1000,3,3)


# In[91]:


predict_price('Whitefield',2000,3,3)


# In[ ]:




