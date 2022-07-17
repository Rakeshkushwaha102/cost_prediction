
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel(r"C:\Users\Radioactive Guy\Downloads\Share data.xlsx")
df.var()
data = pd.read_excel(r"C:\Users\Radioactive Guy\Downloads\Share data.xlsx")
data.head()
data.shape
data.describe()
data.info
data.columns
#### data cleaning
data.duplicated()
data.isnull()
data = data.drop_duplicates()

####EDA
## Data Visualization

#scatter plot of price and course

plt.figure(figsize=(15, 8))
# plot two values price per course
plt.scatter(data.price, data.courses)
plt.xlabel("price ", fontsize=14)
plt.ylabel("courses", fontsize=14)
plt.title("Scatter plot of price and courses",fontsize=18)
plt.show()

#scatter plot of no of hours and price

plt.figure(figsize=(15, 8))
# plot two values price per course
plt.scatter(data.price, data.no_of_hours)
plt.xlabel("price ", fontsize=14)
plt.ylabel("Duration", fontsize=14)
plt.title("Scatter plot of price and no_of_hours",fontsize=18)
plt.show()

## infrastructure vs price
plt.figure(figsize=(15, 8))
# plot two values price per course
plt.scatter(data.price, data.Infrastructure)
plt.xlabel("price ", fontsize=14)
plt.ylabel("Infrastructure", fontsize=14)
plt.title("Scatter plot of price and Infrastructure",fontsize=18)
plt.show()

## teaching staff salary vs price
plt.figure(figsize=(15, 8))
# plot two values price per course
plt.scatter(data.price, data.teaching_staff_salary)
plt.xlabel("price ", fontsize=14)
plt.ylabel("teaching_staff_salary", fontsize=14)
plt.title("Scatter plot of price and teaching_staff_salary",fontsize=18)
plt.show()

## nonteaching staff salary vs price
plt.figure(figsize=(15, 8))
# plot two values price per course
plt.scatter(data.price, data.nonTeaching_staff_salary)
plt.xlabel("price ", fontsize=14)
plt.ylabel("nonTeaching_staff_salary", fontsize=14)
plt.title("Scatter plot of price and nonTeaching_staff_salary",fontsize=18)
plt.show()


###correlation
plt.figure(figsize=(30,9))
sns.heatmap(data.corr(),annot=True)

sns.pairplot(data)
data.corr()

df1 = data.corr()
### Visualizing categorical data
## course,mode of course, level of course, location

plt.hist(data.courses)
plt.hist(data.mode_of_course)

### Data Preparation

#### visualize the data for outliers

plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Edtech Price Distribution Plot')
sns.distplot(data.price)

plt.subplot(1,2,2)
plt.title("Edtech price distribution")
sns.boxplot(y=data.price)

y = data['price']
data.drop(['location',"institute",'teaching_staff_quali','teaching_staff_exp','nonTeaching_staff_exp','price'],axis=1, inplace=True)

num_cols = [col for col in data.columns if data[col].dtype in ['float64','int64']]


### outliers in num_cols

fig = plt.figure(figsize=(15,18))
for index,col in enumerate(data[num_cols]):
    plt.subplot(6,3,index+1)
    sns.boxplot(y=col, data=data[num_cols].dropna())
    plt.grid()
fig.tight_layout(pad=3)

## outliers in rent,teaching staff salary, expenditure of maintanence

## outliers treatment
## rent
IQR = data['rent'].quantile(0.75) - data['rent'].quantile(0.25)
print (IQR)
lower_limit = data['rent'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['rent'].quantile(0.75) + (IQR * 1.5)

data['rent_replaced'] = pd.DataFrame(np.where(data['rent'] > upper_limit, upper_limit, np.where(data['rent'] < lower_limit, lower_limit, data['rent'])))
print(data['rent_replaced'])
sns.boxplot(data.rent_replaced)
data.drop(['rent'], axis=1, inplace=True)

## teaching staff salary
IQR = data['teaching_staff_salary'].quantile(0.75) - data['teaching_staff_salary'].quantile(0.25)
print (IQR)
lower_limit = data['teaching_staff_salary'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['teaching_staff_salary'].quantile(0.75) + (IQR * 1.5)

data['teaching_staff_salary_replaced'] = pd.DataFrame(np.where(data['teaching_staff_salary'] > upper_limit, upper_limit, np.where(data['teaching_staff_salary'] < lower_limit, lower_limit, data['teaching_staff_salary'])))
print(data['teaching_staff_salary_replaced'])
sns.boxplot(data.teaching_staff_salary_replaced)
data.drop(['teaching_staff_salary'], axis=1, inplace=True)

## expenditure of maintanence
IQR = data['expenditure_of_maintanence'].quantile(0.75) - data['expenditure_of_maintanence'].quantile(0.25)
print (IQR)
lower_limit = data['expenditure_of_maintanence'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['expenditure_of_maintanence'].quantile(0.75) + (IQR * 1.5)

data['expenditure_of_maintanence_replaced'] = pd.DataFrame(np.where(data['expenditure_of_maintanence'] > upper_limit, upper_limit, np.where(data['expenditure_of_maintanence'] < lower_limit, lower_limit, data['expenditure_of_maintanence'])))
print(data['expenditure_of_maintanence_replaced'])
sns.boxplot(data.expenditure_of_maintanence_replaced)
data.drop(['expenditure_of_maintanence'], axis=1, inplace=True)



###### standartization and Dummy variables

num_cols = [col for col in data.columns if data[col].dtype in ['float64','int64']]
cat_cols = [col for col in data.columns if data[col].dtype not in ['float64','int64']]

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
data[num_cols] = MinMaxScaler().fit_transform(data[num_cols])
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(data[cat_cols])
encoded_cols = list(encoder.get_feature_names(cat_cols))
data[encoded_cols] = encoder.transform(data[cat_cols])
data.drop(['mode_of_course','webaccess','internship_certificates','courses','level_of_course','books_supplies'],axis=1,inplace=True)
data.rename(columns={'courses_Data Science' : "courses_Data_Science"},inplace=True)
for i in range(0,25):
    print(data.columns[i],",")

####

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x_train, x_test, y_train, y_test = train_test_split(data,y, test_size=0.3,random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score

model = LinearRegression()
model.fit(x_train,y_train)

## train data
pred_train = model.predict(x_train)

r2_train = r2_score(y_train, pred_train)
mse_train = mean_squared_error(y_train, pred_train)
rmse_train = np.sqrt(mse_train)

# Predict on test data
pred_test = model.predict(x_test)

r2_test = r2_score(y_test, pred_test)
mse_test = mean_squared_error(y_test, pred_test)
rmse_test = np.sqrt(mse_test)


################
import pickle
# open a file, where you ant to store the data
pickle_out = open('model.pkl', 'wb')

# dump information to that file
pickle.dump(model, pickle_out)
pickle_out.close()











Infrastructure_max=max(df["Infrastructure"])
internet_Electricity_max=max(df["internet_Electricity"])
nonTeaching_staff_salary=max(df["nonTeaching_staff_salary"])
marketing_expenses=max(df["marketing_expenses"])
no_of_hours=max(df["no_of_hours"])
rent_replaced=max(df["rent"])
teaching_staff_salary=max(df["teaching_staff_salary"])
expenditure_of_maintanence=max(df["expenditure_of_maintanence"])


min(df["Infrastructure"])
min(df["internet_Electricity"])
min(df["nonTeaching_staff_salary"])
min(df["marketing_expenses"])
min(df["no_of_hours"])
min(df["rent"])
min(df["teaching_staff_salary"])
min(df["expenditure_of_maintanence"])










