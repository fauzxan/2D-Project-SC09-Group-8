# 2D-Project-SC09-Group-8
## Cloning:
#### You can clone this project on git bash using the command:
```
$ git config --global http.sslverify "false"
$ git clone https://github.com/fauzxan/2D-Project-SC09-Group-8.git
```
#### You may then proceed to execute commands as was done in the mini projects
## Local computer
#### Note: To run on local computer, you must install gitbash and clone my repository onto your local computer. Then using command prompt, navigate to the location where this folder is stored and then run the command:
```
set FLASK_APP=fl.py
flask run
```

## Teammates

Zhang Chunjie - 1005604 Seetoh Jian Qing -1005600 Mohammed Fauzaan - 1005404 Tan

Kay Wee - 1005468

# Part 1

## Data

## **Dataset Choice (refer to Annex 1.1 for code)**

Our chosen dataset consists of Covid-19 related information from all the countries around

the world from the beginning of the pandemic, 25 February 2020 to 8th November 2021

(Because we extracted this dataset on the 8th of November 2021). This dataset was chosen

because of the large amount of data points, containing 135,577 rows and 67 columns of

data. Furthermore, it contained the independent and dependent variables that we were

interested in studying, and the data for our chosen variables was relatively complete.

The dataset contained enough variables for us to explore. 65 columns of candidate

variables meant that we could try out numerous variations of relationships between data

sets, and in doing so we were able to find relations that we wanted to focus on.

In addition, the data set also contained both categorical data (such as number of female

smokers, male smokers, or the number of persons aged above 70) and continuous data

(such as new cases, deaths, etc). Hence, we were able to play with different types of models.

## **Data Preparaton (refer to Annex 1.2 for code)**

We did the following to clean up and improve the large amount of data.

#### \1. **Cleaning up "NaN" values (unavailable data):** 

The original data consisted of large
slices of ‘NaN’ values indicating that the data was unavailable. This could be due to
multiple factors-the dataset we used showed pandemic data before the creation of
the vaccine. There were also countries like Bangladesh where there were no
effective data collection method in place(hard to collect accurate data for entire
countries due to urban vs rural areas). Therefore, we removed the rows where there
were NaN values in any of the columns.

#### \2. **Choosing a period for our dataset that has a higher accuracy:** In order to make

the data more accurate we made sure to only consider the data from the 180 days
preceding 8th November 2021. We choose this period as it was when the vaccine
was well circulated to most of the countries in Asia. Prior to that period, there was
low access to vaccines for all the countries around the world. By the last 180 days,
most countries have already chosen a strategy to reduce the number of deaths. The
selected period allowed us to calculate and establish a concrete relationship
between the variables under consideration. (Most of which involve vaccination as a
major component).

#### \3. **Sufficient datapoints for our model:** 
It is a general rule of thumb that more data
results in greater accuracy of the resultant model. Therefore, we decided to take in
all the countries in Asia. We could have gone on to include all the countries from
more than one continent, or perhaps even the whole world. We decided to stick to
only one continent due to homogeneity in terms of cultural landscape, and laws
imposed. In addition, Asian countries constitute more than half the world’s
population.
*The dataset we used, after cleaning up the data, consists of 4576 rows of data and 9 columns.*

## **Choice of Variables**

Our chosen metric to evaluate the accuracy of our models is the adjusted R-squared value.
Adjusted R-squared value is chosen since we are trying to build a multiple linear regression
model to predict the number of deaths. Hence, metrics that measure how linearly related
our values are would be the most suitable to determine the accuracy of our model.
Furthermore, since we are dealing with multiple predictors, it is more appropraite to use
adjusted R-squared value rather than R-squared value.

## **First Atempt**

**x variables:** New Daily Vaccinations per Individual, Median Age **y variable:** New Deaths
**Adjusted R^2: 0.03822**
Our x variables were Median Age and New Daily Vaccinations per Individual. Our
hypothesis was that there would be a positive correlation between countries with a higher
median age and the number of new deaths at any specific period of time. However, our
adjusted R-Squared value was very poor. This means that there is little correlation between
the chosen x variables and the number of new deaths in a country.
While discussing as a group, we realised that while age does make a difference in the
fatality of the virus, median age was a mediocre variable. Countries with a lower median
age tend to be less developed. This affects their access to vaccinations. Hence, while the
population of less developed countries could be relatively young, the number of deaths
could still be high.

## **Second Atempt**

**x variables:** New Vaccinations, New Cases **y variable:** New Deaths **Adjusted R^2:**
**0.73923362**
We then considered other variables that might have a stronger correlation to the number
of new deaths. The new x variables gave us a much higher adjusted R^2 value. This shows
us that the number of new cases affected the number of new deaths, as with an increased
number of new vaccinations.

## **Third Atempt**

**x variables:** Average Vaccinations per Individual, New Cases **y variable:** New Deaths **New**
**Adjusted R^2: 0.742771149**
We improved on our model by changing New Daily Vaccinations per Individual to Average
Vaccinations per Individual. The number of new vaccinations are likely to go down in the
long run once a country has a large population of their citizens fully vaccinated. Total
Vaccinations/Individual is a variable that would increase and stay constant in cases of
countries with a high access to vaccines. This should allow our output of new deaths to be
more stable. In short, as long as Total Vaccinations/Individual > 1, that should represent a
population that has at least 1 jab per person.
**Relatonship between Average Vaccinatons per Individual and New Deaths**

We can see that New Deaths were extremely high when the Average Vaccinations per
Individual was under 0.5. As more people got vaccinated, the curve started to drop. When
Average Vaccinations per Individual increased to above 1, the curve started to flatten. This
showed the efficacy of the vaccine once everyone in the country has had at least 1 dosage of
the vaccine.

# **Relatonship between New Cases and New Deaths**

We see a relationship that resembles a logarithimic curve. As New Cases increase, New
Deaths increase. However, at a certain point, the number of New Deaths started to plateau
even while New Cases started to increase.
We think this is because of the effect of vaccination taking place. There will be more and
more asymptomatic Covid cases that does not suffer any serious symptoms that would
otherwise lead to death.





**Conclusion**

Our third model is the best at predicting Covid-19 deaths in various countries as it has the

highest adjusted R-squared value among our three models, which corresponds to having

the strongest linear correlation. From building the model, we obtained the following

coefficient values and hence, final equation:




# **Annex**

## **1.1: Pulling Data(Selectng Columns)**

```
import pandas as pd 

df=pd.read\_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv',dtype='unicode')

#these are the columns under consideration :

columns\_from\_main=['location','date','total\_cases','new\_cases','new\_de

aths','new\_vaccinations','median\_age','population','total\_vaccinations

'] 

print(df)
```


```
iso\_code continent     location        date total\_cases 

new\_cases  \

0           AFG      Asia  Afghanistan  2020-02-24         5.0       

5.0   

1           AFG      Asia  Afghanistan  2020-02-25         5.0       

0.0   





2           AFG      Asia  Afghanistan  2020-02-26         5.0       

0.0   

3           AFG      Asia  Afghanistan  2020-02-27         5.0       

0.0   

4           AFG      Asia  Afghanistan  2020-02-28         5.0       

0.0   

...         ...       ...          ...         ...         ...       .

..   

135789      ZWE    Africa     Zimbabwe  2021-11-21    133647.0      

32.0   

135790      ZWE    Africa     Zimbabwe  2021-11-22    133674.0      

27.0   

135791      ZWE    Africa     Zimbabwe  2021-11-23    133674.0       

0.0   

135792      ZWE    Africa     Zimbabwe  2021-11-24    133747.0      

73.0   

135793      ZWE    Africa     Zimbabwe  2021-11-25    133774.0      

27.0   

`       `new\_cases\_smoothed total\_deaths new\_deaths new\_deaths\_smoothed 

...  \

0                     NaN          NaN        NaN                 NaN 

...   

1                     NaN          NaN        NaN                 NaN 

...   

2                     NaN          NaN        NaN                 NaN 

...   

3                     NaN          NaN        NaN                 NaN 

...   

4                     NaN          NaN        NaN                 NaN 

...   

...                   ...          ...        ...                 ... 

...   

135789             31.286       4699.0        0.0               0.429 

...   

135790             33.714       4699.0        0.0               0.286 

...   

135791             24.143       4699.0        0.0               0.143 

...   

135792             27.143       4703.0        4.0               0.571 

...   

135793             25.857       4704.0        1.0               0.714 

...   

`       `female\_smokers male\_smokers handwashing\_facilities  \

0                 NaN          NaN                 37.746   

1                 NaN          NaN                 37.746   

2                 NaN          NaN                 37.746   

3                 NaN          NaN                 37.746   

4                 NaN          NaN                 37.746   





...               ...          ...                    ...   

135789            1.6         30.7                 36.791   

135790            1.6         30.7                 36.791   

135791            1.6         30.7                 36.791   

135792            1.6         30.7                 36.791   

135793            1.6         30.7                 36.791   

`       `hospital\_beds\_per\_thousand life\_expectancy 

human\_development\_index  \

0                             0.5           64.83                   

0.511   

1                             0.5           64.83                   

0.511   

2                             0.5           64.83                   

0.511   

3                             0.5           64.83                   

0.511   

4                             0.5           64.83                   

0.511   

...                           ...             ...                     

...   

135789                        1.7           61.49                   

0.571   

135790                        1.7           61.49                   

0.571   

135791                        1.7           61.49                   

0.571   

135792                        1.7           61.49                   

0.571   

135793                        1.7           61.49                   

0.571   

`       `excess\_mortality\_cumulative\_absolute 

excess\_mortality\_cumulative  \

0                                       NaN                         

NaN   

1                                       NaN                         

NaN   

2                                       NaN                         

NaN   

3                                       NaN                         

NaN   

4                                       NaN                         

NaN   

...                                     ...                         ..

.   

135789                                  NaN                         

NaN   

135790                                  NaN                         

NaN   





135791                                  NaN                         

NaN   

135792                                  NaN                         

NaN   

135793                                  NaN                         

NaN   

`       `excess\_mortality excess\_mortality\_cumulative\_per\_million  

0                   NaN                                     NaN  

1                   NaN                                     NaN  

2                   NaN                                     NaN  

3                   NaN                                     NaN  

4                   NaN                                     NaN  

...                 ...                                     ...  

135789              NaN                                     NaN  

135790              NaN                                     NaN  

135791              NaN                                     NaN  

135792              NaN                                     NaN  

135793              NaN                                     NaN  

[135794 rows x 67 columns]
```

## **1.2: Preparing Data (Selectng date, Changing to Integer, Removing Null)**

this is to extract all the that fall within the date range of 13th May 2021 to 8th November 2021, i.e., 180 days

```
df['date']=pd.to\_datetime(df['date'])

mask=(df['date']>='5/13/2021')&(df['date']<='11/8/2021')

df\_location=df.loc[(mask)&(df['continent']=='Asia'),columns\_from\_main]

#convert the string value to a number:

df\_location["new\_deaths"]=pd.to\_numeric(df\_location["new\_deaths"], 

downcast='integer')

df\_location["new\_cases"]=pd.to\_numeric(df\_location["new\_cases"], 

downcast='integer')

df\_location['new\_vaccinations']=pd.to\_numeric(df\_location['new\_vaccina

tions'], downcast='integer')

df\_location['population']=pd.to\_numeric(df\_location['population'], 

downcast='integer')

df\_location['median\_age']=pd.to\_numeric(df\_location['median\_age'], 

downcast='integer')

df\_location['total\_vaccinations']=pd.to\_numeric(df\_location['total\_vac

cinations'], downcast='integer')

#This block removes the NaN slices from the excel sheet

df\_location['total\_vaccinations/population']=df\_location['total\_vaccin

ations']/df\_location['population']

df\_location.dropna(subset=['new\_vaccinations'],inplace=True)

df\_location.dropna(subset=['total\_cases'],inplace=True)

df\_location.dropna(subset=['new\_deaths'],inplace=True)

df\_location.dropna(subset=['new\_cases'],inplace=True)

print(len(df\_location))#number of columns

print(df\_location.size)#number of columns \* number of rows (8929\*9)

print("The data under observation are:\n",df\_location)

df\_location.to\_csv('new data.csv',index=False)#this line will save the

new data into your current folder. This was the final excel sheet that

we worked with.

#Further visualizations are done on this file
```

4576

45760
```

The data under observation are:

`            `location       date total\_cases  new\_cases  new\_deaths  \

458     Afghanistan 2021-05-27     68366.0      623.0        14.0   

465     Afghanistan 2021-06-03     75119.0     1093.0        27.0   

8728     Azerbaijan 2021-05-13    328668.0      509.0        16.0   

8729     Azerbaijan 2021-05-14    328994.0      326.0        12.0   

8730     Azerbaijan 2021-05-15    329371.0      377.0        14.0   

...             ...        ...         ...        ...         ...   

133020      Vietnam 2021-10-31    921122.0     5519.0        53.0   

133021      Vietnam 2021-11-01    926720.0     5598.0        48.0   

133022      Vietnam 2021-11-02    932357.0     5637.0        74.0   

133023      Vietnam 2021-11-03    939463.0     7106.0        78.0   

133024      Vietnam 2021-11-04    946043.0     6580.0        59.0   

`        `new\_vaccinations  median\_age  population  

total\_vaccinations  \

458               2859.0        18.6  39835428.0            593313.0  

465               4015.0        18.6  39835428.0            630305.0  

8728             21015.0        32.4  10223344.0           1748525.0  

8729             13107.0        32.4  10223344.0           1761632.0  

8730             15794.0        32.4  10223344.0           1777426.0  

...                  ...         ...         ...                 ...  

133020          554499.0        32.6  98168829.0          81929875.0  

133021         1201589.0        32.6  98168829.0          83131464.0  

133022          959435.0        32.6  98168829.0          84090899.0  

133023          793175.0        32.6  98168829.0          84884074.0  





133024         1435734.0        32.6  98168829.0          86319808.0  

`        `total\_vaccinations/population  

458                          0.014894  

465                          0.015823  

8728                         0.171033  

8729                         0.172315  

8730                         0.173860  

...                               ...  

133020                       0.834581  

133021                       0.846821  

133022                       0.856595  

133023                       0.864674  

133024                       0.879300  

[4576 rows x 10 columns]
```

# **Part 2**

## **Overview About the Problem**

Our study will focus on the relationship between the total vaccination rate in a given
country and the number of new daily cases. While some countries follow the trend of the
higher the total vaccination rate, the lower the number of new daily cases, our study of
some countries showed that this is not always the case. An analysis of this relationship will
aid countries in overcoming the current situation and possibly help in averting similar
situations in the future.
We believe there is no one size fit all approach in data analytics - different countries with
different approaches would produce different results. By building two separate models for
two countries with different approaches, we can better understand why the relationships
between the selected variables for the two models are as such.

## **Dataset**

We did the following to clean up and improve the vast clutter of data.
The model, after cleaning up the data consists of 4576 rows of data and 9 columns. We
wanted to study the relationship between the number of new cases and vaccination (as the
x variable), and the number of deaths (as the y variable).

```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import math

#Import data from website

df=pd.read\_csv('https://covid.ourworldindata.org/data/owid-covid-

data.csv',dtype='unicode')

#Columns we are interested in 

columns\_from\_main=['location','date','new\_cases','total\_vaccinations',

"population"]

#Set the range of date to last 180 days

df['date']=pd.to\_datetime(df['date'])

mask=(df['date']>='5/13/2021')&(df['date']<='11/8/2021')

#Set the location, i.e. which country we are modeling 

location = "Japan"

df\_location=df.loc[(mask)&(df['location']==location),columns\_from\_main

]

#Get rid of NA values 

df\_location.dropna(subset=['total\_vaccinations'],inplace=True)

df\_location.dropna(subset=['population'],inplace=True)

df\_location.dropna(subset=['new\_cases'],inplace=True)

#convert the string value to a number:

df\_location["new\_cases"]=pd.to\_numeric(df\_location["new\_cases"], 

downcast='integer')

df\_location['total\_vaccinations']=pd.to\_numeric(df\_location['total\_vac

cinations'], downcast='integer')

df\_location['population']=pd.to\_numeric(df\_location['population'], 

downcast='integer')

#Data

print(df\_location.dtypes)

mask=(df\_location["new\_cases"] > 0)&(df\_location['total\_vaccinations']

\> 0)

df\_location=df\_location.loc[mask,columns\_from\_main]

print(df\_location.shape)

print("The locations under observation are:\n",df\_location)

```
```
location                      object

date                  datetime64[ns]

new\_cases                      int16

total\_vaccinations             int32

population                     int32

dtype: object

(144, 5)

The locations under observation are:

`       `location       date  new\_cases  total\_vaccinations  population

62196    Japan 2021-05-13       6874             6695597   126050796

62197    Japan 2021-05-14       6262             7116487   126050796





62198    Japan 2021-05-15       6417             7287316   126050796

62199    Japan 2021-05-16       5256             7495054   126050796

62200    Japan 2021-05-17       3677             8102078   126050796

...        ...        ...        ...                 ...         ...

62368    Japan 2021-11-01         84           190022071   126050796

62370    Japan 2021-11-03        264           190595761   126050796

62371    Japan 2021-11-04        158           191044946   126050796

62374    Japan 2021-11-07        157           192078918   126050796

62375    Japan 2021-11-08        100           192610138   126050796

[144 rows x 5 columns]
```

## **Features and Target Preparaton**

Feature: Average Vaccinations per Individual Target: Number of New Daily Cases

We wanted to study the effectiveness of vaccines in preventing people from getting Covid-

\19. This is so that countries are better aware of the effects of having a more vaccinated

population.
```
\# put Python code to prepare your featuers and target

**def** get\_features\_targets(df, feature\_names, target\_names):

`    `df\_feature=df[feature\_names]

`    `df\_target=df[target\_names]

`    `**return** df\_feature, df\_target

**def** normalize\_z(df):

`    `**return** (df-df.mean())/df.std()

**def** prepare\_feature(df\_feature):

`    `df\_feature=df\_feature.to\_numpy()

`    `c\_ones=np.ones((df\_feature.shape[0],1))

`    `df\_feature=np.hstack((c\_ones,df\_feature))

`    `**return** df\_feature

**def** prepare\_target(df\_target):

`    `df\_target=df\_target.to\_numpy()

`    `**return** df\_target

**def** predict(df\_feature, beta):

`    `X=prepare\_feature(normalize\_z(df\_feature))

`    `**return** predict\_norm(X,beta)

**def** predict\_norm(X, beta):

`    `**return** np.matmul(X,beta)

**def** split\_data(df\_feature, df\_target, random\_state=None, 

test\_size=0.5):

`    `indexes=df\_feature.index

`    `**if** random\_state!=None:





`        `np.random.seed(random\_state)

`    `num\_rows = len(indexes)

`    `k = int(test\_size \* num\_rows)

`    `test\_indices = np.random.choice(indexes, k, replace = False)

`    `train\_indices = set(indexes) - set(test\_indices)



`    `df\_feature\_train = df\_feature.loc[train\_indices, :]

`    `df\_feature\_test = df\_feature.loc[test\_indices, :]

`    `df\_target\_train = df\_target.loc[train\_indices, :]

`    `df\_target\_test = df\_target.loc[test\_indices, :]

`    `**return** df\_feature\_train, df\_feature\_test, df\_target\_train, 

df\_target\_test



**def** r2\_score(y, ypred):

`    `actual\_mean = np.mean(y)

`    `# since y, ypred are both nparray, y-ypred does element wise SUB

`    `ssres = np.sum((y-ypred)\*\*2)

`    `sstot = np.sum((y-actual\_mean)\*\*2)

`    `**return** 1 - (ssres/sstot)

**def** mean\_squared\_error(target, pred):

`    `num\_of\_samples = target.shape[0] # number of samples == number of 

rows in target\_y

`    `**return** (1/num\_of\_samples) \* np.sum((target-pred)\*\*2)

**def** gradient\_descent(X, y, beta, alpha, num\_iters):

`    `# For linreg, X is a n by 2 matrix, beta is a 2 by 1 vector, y 

(actual\_target) n by 1 vector, alpha is Float, num\_iters is an Int

`    `# beta -> initial guess of beta 

`    `### BEGIN SOLUTION

`    `number\_of\_samples = X.shape[0]

`    `J\_storage = np.zeros((num\_iters, 1))

`    `# iterate the grad desc until num\_iters

`    `# or, until convergence (other case)

`    `**for** i **in** range(num\_iters):

`        `# this derivate is derived from the squared error function 

`        `# STEP 2

`        `# Y\_pred = X x Beta

`        `# diff Y\_pred/Beta --> X.T x (X x Beta) 

`        `# transpose X and put on the left hand side of matrix mul

`        `derivative\_cost\_wrt\_Beta = (1/number\_of\_samples) \* 

np.matmul(X.T, (np.matmul(X, beta) - y))

`        `# update beta

`        `# STEP 3

`        `beta = beta - alpha \* derivative\_cost\_wrt\_Beta

`        `J\_storage[i] = compute\_cost(X, y, beta)

`    `### END SOLUTION

`    `**return** beta, J\_storage

**def** compute\_cost(X, y, beta):





`    `# for LinReg: X is n by 2, y is a vector of n elements, beta is 2 

by 1

`    `J = 0

`    `### BEGIN SOLUTION

`    `number\_of\_samples = X.shape[0]

`    `# Y\_pred - Y\_actual

`    `# Y\_pred = Xb

`    `Y\_pred = np.matmul(X, beta)

`    `diff\_between\_pred\_actual\_y = Y\_pred - y

`    `diff\_between\_pred\_actual\_y\_sq = 

np.matmul(diff\_between\_pred\_actual\_y.T, diff\_between\_pred\_actual\_y)

`    `J = (1/(2\*number\_of\_samples)) \* diff\_between\_pred\_actual\_y\_sq

`    `### END SOLUTION

`    `# J is an error, it is a scalar, so extract the only element of J 

that was a numpy array

`    `**return** J[0][0]

**def** logarithm(df\_target, target\_name):

`    `### BEGIN SOLUTION

`    `df\_out = df\_target.copy()

`    `df\_out.loc[:, target\_name] = df\_target[target\_name].apply(**lambda** 

x: math.log(x))

`    `### END SOLUTION

`    `**return** df\_out

#Create a new column to contain the data for 

total\_vaccinations/population

df\_location.loc[:,"total\_vaccinations/population"] = 

df\_location['total\_vaccinations'].div(df\_location['population'])

#Set features and target

features=['total\_vaccinations/population']

target=['new\_cases']

df\_features,df\_target=get\_features\_targets(df\_location,features,target

)

#myplot = sns.scatterplot(x="total\_vaccinations/population", 

y="new\_cases", data=df\_location)

#Split data into two groups for training and testing

df\_features\_train, df\_features\_test, df\_target\_train, df\_target\_test =

split\_data(df\_features,df\_target,random\_state=100,test\_size=0.3)

#Normalize both train and test features

df\_features\_train = normalize\_z(df\_features\_train)

df\_features\_test = normalize\_z(df\_features\_test)

X=prepare\_feature(df\_features\_train)

target=prepare\_target(df\_target\_train)
```




## **Building Model**

We used Linear Regression to model the relationship between our chosen feature and
target. We found differing relationships between the total vaccination rate and the number
of new daily cases in different countries. Our hypothesis for this is that it is due to the
different rates at which each country had gotten their population vaccinated.
We will be looking at 2 categories of countries:
\1. countries that had achieved a high rate of vaccination early on in Covid-19
\2. countries that were slower in vaccinating their population against Covid-19
For the first category, we will be looking at the case study of Japan and for the second
category, we will be looking at India.
*note: all x values has been normalized for the construction of the model*


### **Model 1: Japan**

```
\# model for Japan

iterations=1500

alpha=0.01

beta=np.zeros((2,1))

beta,J\_storage=gradient\_descent(X,target,beta,alpha,iterations)

pred=predict(df\_features\_test,beta)

print("Beta0 and beta1 is equal to: ")

print(beta)

plt.scatter(df\_features\_test, df\_target\_test)

plt.plot(df\_features\_test, pred, color="orange")

Beta0 and beta1 is equal to: 

[[ 59800.81774788]

` `[-42139.46143839]]

[<matplotlib.lines.Line2D at 0x7f9229569b90>]
```


### *Evaluatng the Model*

#### **Horizontal Axis:** Average Vaccinations per Individual

#### **Vertical Axis:** New Cases

When we took the entire period of 180 days, the R-squared value was low at 0.0009655.
The explanation for this lies in their approach to Covid-19 at different points in time. There
is an observable decrease in cases at the start, due to Japan implementing national
regulations dictating how their citizens needed to stay indoors. Japan then removed their
regulations for the Olympics and new cases skyrocketed. However, as more Japanese got
vaccinated, the number of new cases started going down again.

```
the\_target = prepare\_target(df\_target\_test)

r2= r2\_score(the\_target, pred)

print("R2 Coefficient of Determination:")

print(r2)

mse = mean\_squared\_error(the\_target, pred)

print("Mean Squared Error:")

print(mse)

R2 Coefficient of Determination:

0.47027739501595534

Mean Squared Error:

2357072263.0297494

**Model 2: India**

df=pd.read\_csv('https://covid.ourworldindata.org/data/owid-covid-

data.csv',dtype='unicode')





#Columns we are interested in 

columns\_from\_main=['location','date','new\_cases','total\_vaccinations',

"population"]

#Set the range of date to last 180 days

df['date']=pd.to\_datetime(df['date'])

mask=(df['date']>='5/13/2021')&(df['date']<='11/8/2021')

#Set the location, i.e. which country we are modeling 

location = "India"

df\_location=df.loc[(mask)&(df['location']==location),columns\_from\_main

]

#Get rid of NA values 

df\_location.dropna(subset=['total\_vaccinations'],inplace=True)

df\_location.dropna(subset=['population'],inplace=True)

df\_location.dropna(subset=['new\_cases'],inplace=True)

#convert the string value to a number:

df\_location["new\_cases"]=pd.to\_numeric(df\_location["new\_cases"], 

downcast='integer')

df\_location['total\_vaccinations']=pd.to\_numeric(df\_location['total\_vac

cinations'], downcast='integer')

df\_location['population']=pd.to\_numeric(df\_location['population'], 

downcast='integer')

#Data

mask=(df\_location["new\_cases"] > 0)&(df\_location['total\_vaccinations']

\> 0)

df\_location=df\_location.loc[mask,columns\_from\_main]

#Create a new column to contain the data for 

total\_vaccinations/population

df\_location.loc[:,"total\_vaccinations/population"] = 

df\_location['total\_vaccinations'].div(df\_location['population'])

#Set features and target

features=['total\_vaccinations/population']

target=['new\_cases']

df\_features,df\_target=get\_features\_targets(df\_location,features,target

)

#myplot = sns.scatterplot(x="total\_vaccinations/population", 

y="new\_cases", data=df\_location)

#Split data into two groups for training and testing

df\_features\_train, df\_features\_test, df\_target\_train, df\_target\_test =

split\_data(df\_features,df\_target,random\_state=100,test\_size=0.3)





#Normalize both train and test features

df\_features\_train = normalize\_z(df\_features\_train)

df\_features\_test = normalize\_z(df\_features\_test)

X=prepare\_feature(df\_features\_train)

target=prepare\_target(df\_target\_train)

iterations=1500

alpha=0.01

beta=np.zeros((2,1))

beta,J\_storage=gradient\_descent(X,target,beta,alpha,iterations)

pred=predict(df\_features\_test,beta)

print("Beta0 and beta1 is equal to: ")

print(beta)

plt.scatter(df\_features\_test, df\_target\_test)

plt.plot(df\_features\_test, pred, color="orange")

Beta0 and beta1 is equal to: 

[[ 59800.81774788]

` `[-42139.46143839]]

[<matplotlib.lines.Line2D at 0x7f9227366cd0>]
```

**Horizontal Axis:** Total Vaccinations/Individual
**Vertical Axis:** New Cases
Our R-squared value was 0.47028, which indicates that total vaccinations per individual
and new cases do not have a strong linear relationship.


Even though the Average Vaccinations per Individual remains much lower than Japan, their
new cases dropped drastically. After researching about India's approach, our conclusion is
that India had achieved a higher herd immunity than other countries with a high
vaccination rate like Japan. A study involving 29 000 participants from 70 districts across
India found that 67.6% of them had Covid antibodies. In Delhi, a serology test showed 97%
of their citizens had Covid antibodies. This herd immunity explains why even though their
vaccination rate was low, their daily cases remained relatively stable.

## **Improving the Model**

*Model 1: Japan*

Features and Target Preparatin

```
#Import data from website

df=pd.read\_csv('https://covid.ourworldindata.org/data/owid-covid-

data.csv',dtype='unicode')

#Columns we are interested in 

columns\_from\_main=['location','date','new\_cases','total\_vaccinations',

"population"]

#Set the range of date to last 180 days

df['date']=pd.to\_datetime(df['date'])

mask=(df['date']>='5/13/2021')&(df['date']<='11/8/2021')

#Set the location, i.e. which country we are modeling 

location = "Japan"

df\_location=df.loc[(mask)&(df['location']==location),columns\_from\_main

]

#Get rid of NA values 

df\_location.dropna(subset=['total\_vaccinations'],inplace=True)

df\_location.dropna(subset=['population'],inplace=True)

df\_location.dropna(subset=['new\_cases'],inplace=True)

#convert the string value to a number:

df\_location["new\_cases"]=pd.to\_numeric(df\_location["new\_cases"], 

downcast='integer')

df\_location['total\_vaccinations']=pd.to\_numeric(df\_location['total\_vac

cinations'], downcast='integer')

df\_location['population']=pd.to\_numeric(df\_location['population'], 

downcast='integer')

#Data

mask=(df\_location["new\_cases"] > 0)&(df\_location['total\_vaccinations']

\> 0)

df\_location=df\_location.loc[mask,columns\_from\_main]





df\_location.loc[:,"total\_vaccinations/population"] = 

df\_location['total\_vaccinations'].div(df\_location['population'])

features=['total\_vaccinations/population']

target=['new\_cases']

df\_features,df\_target=get\_features\_targets(df\_location,features,target

)

\# Try to set the range for x i.e.total\_vaccinations/population

**if** location=='Japan':

`    `df\_features\_improved = 

df\_features.loc[df\_features["total\_vaccinations/population"]>0.8]



#print(df\_location.loc[df\_location["total\_vaccinations"]/df\_location["

population"]>0.8])

**else**:

`    `df\_features\_improved = df\_features

df\_target\_improved = df\_target.loc[set(df\_features\_improved.index),:]

#print(df\_features\_improved)

#myplot = sns.scatterplot(x="total\_vaccinations/population", 

y="new\_cases", data=df\_location)

#Split data

df\_features\_train, df\_features\_test, df\_target\_train, df\_target\_test =

split\_data(df\_features\_improved,df\_target\_improved,random\_state=100,te

st\_size=0.3)

#Apply logarithm to change y to lny

df\_target\_train = logarithm(df\_target\_train, "new\_cases")

df\_target\_test = logarithm(df\_target\_test, "new\_cases")

#print(df\_target\_train)

print(df\_features\_train.mean())

print(df\_features\_train.std())

#Normalize both train and test features

df\_features\_train = normalize\_z(df\_features\_train)

df\_features\_test = normalize\_z(df\_features\_test)

total\_vaccinations/population    1.225269

dtype: float64

total\_vaccinations/population    0.216455

dtype: float64

Building Midel

X=prepare\_feature(df\_features\_test)

Y=prepare\_target(df\_target\_test)

#print(df\_target\_test)

iterations=1500

alpha=0.01

beta=np.zeros((2,1))





beta,J\_storage=gradient\_descent(X,Y,beta,alpha,iterations)

pred=predict(df\_features\_test,beta)

print("Beta0 and beta1 is equal to: ")

print(beta)

Beta0 and beta1 is equal to: 

[[ 7.59424472]

` `[-1.92624804]]

```

## Evaluatng the Model

To obtain a regression that better represents the relationship that we wanted to study, we
isolated the cases starting from New Cases and Average Vaccinations per Individual of >0.8.
The R-squared value was low because it included the abnomaly caused by the Olympics. By
isolating the cases after the event, particularly after the Average Vaccinations per
Individual > 0.8, we can find a model that reflects the number of New Cases in a country
that managed to achieve a high vaccination rate. We also changed the y-value to ln(y). This
was to check our hypothesis that the relationship between New Cases and Average
Vaccinations per Individual was followed a natural logarithmic one.
We ended up with a good regression through this method. This essentially shows the rate
of decrease in the number of New Cases once the number of vaccinations passes a critical
juncture. The resultant R-squared value was relatively high at 0.86413, which shows that
the relationship between the number of New Cases and Average Vaccinations per
Individual is follows a natural logarithmic relationship for countries who achieved high
vaccination rates at an earlier stage of Covid-19.

```
plt.scatter(df\_features\_test, df\_target\_test)

plt.plot(df\_features\_test, pred, color="orange")

the\_target = prepare\_target(df\_target\_test)

r2= r2\_score(the\_target, pred)

print("R2 Coefficient of Determination:")

print(r2)

mse = mean\_squared\_error(the\_target, pred)

print("Mean Squared Error:")

print(mse)
```
```

R2 Coefficient of Determination:

0.8888272902263484

Mean Squared Error:

0.43679422870053336

```



## **Improving the Model**

*Model 2: India*

```

#Import data from website

df=pd.read\_csv('https://covid.ourworldindata.org/data/owid-covid-

data.csv',dtype='unicode')

#Columns we are interested in 

columns\_from\_main=['location','date','new\_cases','total\_vaccinations',

"population"]

#Set the range of date to last 180 days

df['date']=pd.to\_datetime(df['date'])

mask=(df['date']>='5/13/2021')&(df['date']<='11/8/2021')

#Set the location, i.e. which country we are modeling 

location = "India"

df\_location=df.loc[(mask)&(df['location']==location),columns\_from\_main

]

#Get rid of NA values 

df\_location.dropna(subset=['total\_vaccinations'],inplace=True)

df\_location.dropna(subset=['population'],inplace=True)

df\_location.dropna(subset=['new\_cases'],inplace=True)

#convert the string value to a number:

df\_location["new\_cases"]=pd.to\_numeric(df\_location["new\_cases"], 

downcast='integer')

df\_location['total\_vaccinations']=pd.to\_numeric(df\_location['total\_vac





cinations'], downcast='integer')

df\_location['population']=pd.to\_numeric(df\_location['population'], 

downcast='integer')

#Data

mask=(df\_location["new\_cases"] > 0)&(df\_location['total\_vaccinations']

\> 0)

df\_location=df\_location.loc[mask,columns\_from\_main]

df\_location.loc[:,"total\_vaccinations/population"] = 

df\_location['total\_vaccinations'].div(df\_location['population'])

features=['total\_vaccinations/population']

target=['new\_cases']

df\_features,df\_target=get\_features\_targets(df\_location,features,target

)

\# Try to set the range for x i.e.total\_vaccinations/population

**if** location=='Japan':

`    `df\_features\_improved = 

df\_features.loc[df\_features["total\_vaccinations/population"]>0.8]



#print(df\_location.loc[df\_location["total\_vaccinations"]/df\_location["

population"]>0.8])

**else**:

`    `df\_features\_improved = df\_features

df\_target\_improved = df\_target.loc[set(df\_features\_improved.index),:]

#print(df\_features\_improved)

#myplot = sns.scatterplot(x="total\_vaccinations/population", 

y="new\_cases", data=df\_location)

#Split data

df\_features\_train, df\_features\_test, df\_target\_train, df\_target\_test =

split\_data(df\_features\_improved,df\_target\_improved,random\_state=100,te

st\_size=0.3)

#Apply logarithm to change y to lny

df\_target\_train = logarithm(df\_target\_train, "new\_cases")

df\_target\_test = logarithm(df\_target\_test, "new\_cases")

#print(df\_target\_train)

print(df\_features\_train.mean())

print(df\_features\_train.std())

#Normalize both train and test features

df\_features\_train = normalize\_z(df\_features\_train)

df\_features\_test = normalize\_z(df\_features\_test)





total\_vaccinations/population    0.404356

dtype: float64

total\_vaccinations/population    0.205019

dtype: float64

Building Midel

X=prepare\_feature(df\_features\_test)

Y=prepare\_target(df\_target\_test)

#print(df\_target\_test)

iterations=1500

alpha=0.01

beta=np.zeros((2,1))

beta,J\_storage=gradient\_descent(X,Y,beta,alpha,iterations)

pred=predict(df\_features\_test,beta)

print("Beta0 and beta1 is equal to: ")

print(beta)

Beta0 and beta1 is equal to: 

[[10.54177744]

` `[-0.78281369]]

```


## Evaluatng the Model

For India, we also changed the y-value to ln(y) to test for a linear relationship between New
Cases and Average Vaccinations per Individual. The reason why we kept the same cases for
our improved model (as opposed to Japan where we isolated Average Vaccinations per
Individual) is because there is no observable fluctuation in the graph. Instead, a decrease at
an increasing rate is observed. The resultant R-squared value was relatively high at
0.82467, which shows that the relationship between the number of new cases and total
vaccinations/individual follows a natural logarithmic relationship.
```
plt.scatter(df\_features\_test, df\_target\_test)

plt.plot(df\_features\_test, pred, color="orange")

the\_target = prepare\_target(df\_target\_test)

r2= r2\_score(the\_target, pred)

print("R2 Coefficient of Determination:")

print(r2)

mse = mean\_squared\_error(the\_target, pred)

print("Mean Squared Error:")

print(mse)

R2 Coefficient of Determination:

0.8246703587807211

Mean Squared Error:

0.12772971386438003

```



## **Discussion and Analysis**

A country with high vaccination rates early on into the virus should use our Japan model
while a country which is slower in getting their country's vaccination rates up should use
our India model to model the relationship between total vaccination rates and new daily
cases. Our models follow a natural logarithmic relationship, but the 2 models have different
beta values and y-intercepts.

# **References**

ACR recommends booster dose of COVID-19 mrna vaccine for patients on

immunosuppressants. Healio. (n.d.). Retrieved November 26, 2021, from

[https://www.healio.com/news/rheumatology/20210823/acr-recommends-booster-dose-](https://www.healio.com/news/rheumatology/20210823/acr-recommends-booster-dose-of-covid19-mrna-vaccine-for-patients-on-immunosuppressants)

[of-covid19-mrna-vaccine-for-patients-on-immunosuppressants](https://www.healio.com/news/rheumatology/20210823/acr-recommends-booster-dose-of-covid19-mrna-vaccine-for-patients-on-immunosuppressants).





W3.CSS templates. (n.d.). Retrieved November 26, 2021, from

<https://www.w3schools.com/w3css/w3css_templates.asp>.

freeCodeCamp.org. (2020, April 1). How to build a web application using flask and deploy it

to the cloud. freeCodeCamp.org. Retrieved November 26, 2021, from

[https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-](https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/)

[deploy-it-to-the-cloud-3551c985e492/](https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/).

How to display a PDF file in a HTML web page. Techwalla. (n.d.). Retrieved November 26,

2021, from [https://www.techwalla.com/articles/how-to-display-a-pdf-file-in-a-html-web-](https://www.techwalla.com/articles/how-to-display-a-pdf-file-in-a-html-web-page)

[page](https://www.techwalla.com/articles/how-to-display-a-pdf-file-in-a-html-web-page).


