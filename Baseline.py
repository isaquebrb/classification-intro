import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# predict between project finished or unfinished

# csv file containing data about projects, it's prices, expected hours and if it was finished or not
csv_path = "projects.csv"

data = pd.read_csv(csv_path)
print("Data head columns:")
print(data.head())

# lets change unfinished to finished, for simplicity purpose
change = {
    0: 1,
    1: 0
}

# create new column remapping unfinished column
data['finished'] = data.unfinished.map(change)
print("Data new column:")
print(data.tail())

# data visualization library
sns.scatterplot(x="expected_hours", y="price", hue="finished", data=data)

# lets train model with data, create new objects
x = data[['expected_hours', 'price']]
y = data['finished']

seed = 20
# divide between train and test data, with 25%(0.25) for test
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=seed, test_size=0.25, stratify=y)
print("Training data")
print("We trained with %d elements and will test with %d elements" % (len(train_x), len(test_x)))

model = LinearSVC()
model.fit(train_x, train_y)
predict = model.predict(test_x)

# accuracy score low
accuracy = accuracy_score(test_y, predict) * 100
print("The accuracy was %.2f%%" % accuracy)

# fake a predict baseline, set 1 to all tests
baseline_predict = np.ones(540)
baseline_accuracy = accuracy_score(test_y, baseline_predict) * 100
print("The accuracy baseline was %.2f%%" % baseline_accuracy)

