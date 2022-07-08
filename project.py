import pandas as pd
import numpy as np
df = pd.read_excel(r'E:\360 classes\python\project1.xlsx')
df.columns

df.var() ## found 3 zero variance so removed the columns

df.drop(['Unnamed: 0', 'S.No','Total_Course_Hours', 'Licencing_and_Registration','Infrastructure_Cost','Technical_Requirements', 'Institute', 'Institute_brand_value'],axis=1, inplace=True) ## dropping the columns which are not use ful for prediction


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Course_Offered'] = le.fit_transform(df['Course_Offered'])
df['Course_level'] = le.fit_transform(df['Course_level'])
df['No_of_Instructors'] = le.fit_transform(df['No_of_Instructors'])
df['Mode_of_Course'] = le.fit_transform(df['Mode_of_Course'])
df['Webportal_Access'] = le.fit_transform(df['Webportal_Access'])
df['Certification_Exam'] = le.fit_transform(df['Certification_Exam'])
df['Placement_Offered'] = le.fit_transform(df['Placement_Offered'])
df['Level'] = le.fit_transform(df['Level'])

## Exploratory data Analysis:
# 1. Measure of central tendency
# 2. Measure of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distribution of variable
# 6. Graphical representation 

df.describe()

## Q - Q plot
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pylab
stats.probplot(df.Price, dist= 'norm', plot= pylab)
stats.probplot(np.log(df.Price), dist= 'norm', plot=pylab) ## applying log to normal the data

## Graphical representation 
import matplotlib.pyplot as plt ## used for visualization

plt.bar(height= df.Price, x =np.arange(1,151,1)) ## bar plot
plt.hist(df.Price) # histogram
plt.boxplot(df.Price)

plt.boxplot(df['teaching_staff_salary'])
plt.boxplot(df['Monthly_Rent'])

## Scatter plot
plt.scatter(x = df['teaching_staff_salary'], y= df['Price'], color= 'Green')

## correlation
np.corrcoef(x = df['teaching_staff_salary'], y= df['Price'])

import seaborn as sns

sns.pairplot(df.iloc[:, :])
## covariance
# Numpy does not have a function to calculate the covariance between two variables directly
# Function to calculating covariance matrix is called cov()
# By default the cov function will calculate unbiased or sample covariance between the provided random variable

cov_output = np.cov(df['teaching_staff_salary'],  df['Price'])[0,1]

df.cov()

## Import library
import statsmodels.formula.api as smf

## Multilinear regression
model = smf.ols('Price ~Course_Offered+Course_level+No_of_Instructors+Monthly_Rent+Monthly_Bills+Advertising_Marketing+Maintenance+Non_teaching_staff_salary+teaching_staff_salary', data=df).fit()
model.summary()

## Check wheather data has any influence value
# Influence index plot

import statsmodels.api as sm

sm.graphics.influence_plot(model)

## Studentized Residual = Residual /Standard deviation of residual
## index 86 showing high influence so we can exclude the entire row

df1 = df.drop(df.index[[86,95]])

##  Preparing the model 
new_model = smf.ols('Price ~Course_Offered+Course_level+No_of_Instructors+Monthly_Rent+Monthly_Bills+Advertising_Marketing+Maintenance+Non_teaching_staff_salary+teaching_staff_salary', data=df1).fit()

new_model.summary()

## check for collinarity to decide to remove a variable using VIF
## Assumption: VIF > 10 = colinarity 
## calculating VIF values of independent variable


rsq_Course_Offered = smf.ols('Course_Offered ~ Course_level+No_of_Instructors+Monthly_Rent+Monthly_Bills+Advertising_Marketing+Maintenance+Non_teaching_staff_salary+teaching_staff_salary', data=df).fit().rsquared
vif_Course_Offered = 1/(1-rsq_Course_Offered)

rsq_Course_level = smf.ols('Course_level ~ Course_Offered+No_of_Instructors+Monthly_Rent+Monthly_Bills+Advertising_Marketing+Maintenance+Non_teaching_staff_salary+teaching_staff_salary', data=df).fit().rsquared
vif_Course_level= 1/(1- rsq_Course_level)

rsq_No_of_Instructors = smf.ols('No_of_Instructors ~Course_Offered+Course_level+Monthly_Rent+Monthly_Bills+Advertising_Marketing+Maintenance+Non_teaching_staff_salary+teaching_staff_salary', data=df).fit().rsquared
vif_No_of_Instructors = 1/ (1-rsq_No_of_Instructors)

rsq_Monthly_Rent = smf.ols('Monthly_Rent ~ Course_Offered+Course_level+No_of_Instructors+Monthly_Bills+Advertising_Marketing+Maintenance+Non_teaching_staff_salary+teaching_staff_salary', data=df).fit().rsquared
vif_Monthly_Rent = 1/(1- rsq_Monthly_Rent)

rsq_Monthly_Bills = smf.ols('Monthly_Bills ~ Course_Offered+Course_level+No_of_Instructors+Monthly_Rent+Advertising_Marketing+Maintenance+Non_teaching_staff_salary+teaching_staff_salary', data=df).fit().rsquared
vif_Monthly_Bills = 1/(1- rsq_Monthly_Bills)

rsq_Advertising_Marketing =smf.ols('Advertising_Marketing ~ Course_Offered+Course_level+No_of_Instructors+Monthly_Rent+Monthly_Bills+Maintenance+Non_teaching_staff_salary+teaching_staff_salary', data=df).fit().rsquared
vif_Advertising_Marketing = 1/(1- rsq_Advertising_Marketing)

rsq_Maintenance = smf.ols('Maintenance ~ Course_Offered+Course_level+No_of_Instructors+Monthly_Rent+Monthly_Bills+Advertising_Marketing+Non_teaching_staff_salary+teaching_staff_salary', data=df).fit().rsquared
vif_Maintenance = 1/(1- rsq_Maintenance)

rsq_Non_teaching_staff_salary = smf.ols('Non_teaching_staff_salary ~Course_Offered+Course_level+No_of_Instructors+Monthly_Rent+Monthly_Bills+Advertising_Marketing+Maintenance+teaching_staff_salary', data=df).fit().rsquared
vif_Non_teaching_staff_salary = 1/ (1- rsq_Non_teaching_staff_salary)

rsq_teaching_staff_salary = smf.ols('teaching_staff_salary ~ Course_Offered+Course_level+No_of_Instructors+Monthly_Rent+Monthly_Bills+Advertising_Marketing+Maintenance+Non_teaching_staff_salary', data=df).fit().rsquared 
vif_teaching_staff_salary = 1/(1- rsq_teaching_staff_salary)

## Storing vif values in dataframe
new_df = {'Variables': ['Course_Offered', 'Course_level', 'No_of_Instructors', 'Monthly_Rent', 'Monthly_Bills', 'Advertising_Marketing', 'Maintenance', 'Non_teaching_staff_salary', 'teaching_staff_salary'], 'VIF': [vif_Course_Offered,vif_Course_level,vif_No_of_Instructors,vif_Monthly_Rent,vif_Monthly_Bills,vif_Advertising_Marketing,vif_Maintenance,vif_Non_teaching_staff_salary,vif_teaching_staff_salary]}

Vif_dataframe = pd.DataFrame(new_df)
Vif_dataframe

## Removing  Advertising_Marketing as highest collinearity 

## Final model
final_model = smf.ols('Price ~Monthly_Rent+Course_Offered+Course_level+No_of_Instructors+Monthly_Bills+Maintenance+Non_teaching_staff_salary+teaching_staff_salary', data=df).fit()
final_model.summary()

## Prediction
pred = final_model.predict(df)
pred

## Q - Q plol
res = final_model.resid
stats.probplot(res, dist= 'norm', plot=pylab)
plt.show()

## Residuals vs Fitted plot
sns.residplot(x = pred, y = df.Price, lowess= True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Residuals vs Fitted plot')
plt.show()

## Splitting the data into train and test

from sklearn.model_selection import train_test_split 
df_train, df_test = train_test_split(df, test_size=0.20)

## Preparing the model on train data
model_train = smf.ols('Price ~ Monthly_Rent+Course_Offered+Course_level+No_of_Instructors+Monthly_Bills+Maintenance+Non_teaching_staff_salary+teaching_staff_salary', data=df_train).fit()

# Prediction the model on test data
test_pred = model_train.predict(df_test)

## test residual values
test_resid = test_pred - df_test.Price
## RMSE for test data
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

## train data prediction
train_pred = model_train.predict(df_train)

# train residual values
train_resid = train_pred - df_train.Price
## RMSE for the train
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

## Applying Random Forest Algorithmn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel(r'E:\360 classes\python\project1.xlsx')

data.columns
data.duplicated().sum()

data.var() ## found 3 zero variance so removed the columns

data.drop(['Unnamed: 0', 'S.No','Total_Course_Hours', 'Licencing_and_Registration','Infrastructure_Cost','Technical_Requirements', 'Institute', 'Institute_brand_value'],axis=1, inplace=True) ## dropping the columns which are not use ful for prediction


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['Course_Offered'] = le.fit_transform(data['Course_Offered'])
data['Course_level'] = le.fit_transform(data['Course_level'])
data['No_of_Instructors'] = le.fit_transform(data['No_of_Instructors'])
data['Mode_of_Course'] = le.fit_transform(data['Mode_of_Course'])
data['Webportal_Access'] = le.fit_transform(data['Webportal_Access'])
data['Certification_Exam'] = le.fit_transform(data['Certification_Exam'])
data['Placement_Offered'] = le.fit_transform(data['Placement_Offered'])
data['Level'] = le.fit_transform(data['Level'])

data

from sklearn.model_selection import train_test_split
x = data.iloc[:, :14]
y = data.iloc[:, -1:]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.20)  


## Fitting Random forest to the dataset
from sklearn.ensemble import RandomForestRegressor

## creating regression object
regression = RandomForestRegressor(n_estimators=100, n_jobs=1,random_state=0)

## fit the regression with x and y data
regression.fit(x_train,y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# prediction on test data
confusion_matrix(y_test, regression.predict(x_test))
accuracy_score(y_test, regression.predict(x_test))

## prediction on train data
confusion_matrix(y_train, regression.predict(x_train))
accuracy_score(y_train, regression.predict(x_train))



