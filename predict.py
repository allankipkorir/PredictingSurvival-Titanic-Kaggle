import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#survival by sex
sex_pivot = pd.pivot_table(train, index='Sex', values='Survived')
sex_pivot.plot.bar()

# survival by paid class ticket 
paid_class_pivot = pd.pivot_table(train, index='Pclass', values='Survived')
paid_class_pivot.plot.bar()

survived = train[train['Survived']==1]
died = train[train['Survived']==0]

survived['Age'].plot.hist(alpha=0.5, color='blue', bins=50)
died['Age'].plot.hist(alpha=0.5, color='red', bins=50)
plt.legend(['Survived', 'Died'])

# plt.show()

# catagorize the ages

def categorize_ages(df, cut_points, labels):
    df['Age'] = df['Age'].fillna(-0.5)
    df['Age_Category'] = pd.cut(df['Age'], cut_points, labels=labels)
    return df

cut_points = [-1, 0, 5, 12, 18, 35, 60, 120]
labels = ["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior']
train = categorize_ages(train, cut_points, labels)
test = categorize_ages(test, cut_points, labels)

# survival by age category
age_cat_survival = train.pivot_table(index='Age_Category', values='Survived')
age_cat_survival.plot.bar()
# plt.show()

# create dummy columns
def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df

train = create_dummies(train, 'Pclass')
test = create_dummies(test, 'Pclass')

train = create_dummies(train,"Sex")
test = create_dummies(test,"Sex")

train = create_dummies(train,"Age_Category")
test = create_dummies(test,"Age_Category")

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_Category_Missing','Age_Category_Infant',
       'Age_Category_Child', 'Age_Category_Teenager',
       'Age_Category_Young Adult', 'Age_Category_Adult',
       'Age_Category_Senior']

# split the data into train and test data
all_X = train[columns]
all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.2,random_state=0)

 # train the model and predict
lr = LogisticRegression()

lr.fit(train_X, train_y)
prediction = lr.predict(test_X)

# get accuracy
accuracy = accuracy_score(test_y, prediction)
print(accuracy)