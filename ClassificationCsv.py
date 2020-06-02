import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# predict between product bought or not

# csv file containing data about people who bought product and if they accessed a page
csv = "tracking.csv"

# read csv with pandas
data = pd.read_csv(csv)

map = {
    "home": "home_page",
    "how_it_works": "how_it_works_page",
    "contact": "contact_page",
    "bought": "bought_product"
}

# rename columns using map obj
data = data.rename(columns=map)

# x -> page accessed by people, y -> who bough a product
x = data[["home_page", "how_it_works_page", "contact_page"]]
y = data["bought_product"]

# divide data (75 for train and 24 for test)
train_x = x[:75]
train_y = y[:75]
test_x = x[75:]
test_y = y[75:]

print("We trained with %d elements and will test with %d elements" % (len(train_x), len(test_x)))

# train the model with people interactions (train_x) and if they bought a product (train_y)
model = LinearSVC()
model.fit(train_x, train_y)

# predict if they bought given the people interaction (text_x data)
predictions = model.predict(test_x)

# percent of correct predictions according with real case (test_y)
accuracy = accuracy_score(test_y, predictions) * 100
print("The accuracy was %.2f%%" % accuracy)

# using another way to separate train and test, with train_test_split
# lock the random order of the numbers in the test
seed = 20

train2_x, test2_x, train2_y, test2_y = train_test_split(x, y,
                                                        test_size=0.24,
                                                        random_state=seed,
                                                        stratify=y)  # stratify data proportionally according y
print("Using train_test_split, we trained with %d elements and will test with %d elements" %
      (len(train2_x), len(test2_x)))

model2 = LinearSVC()
model2.fit(train2_x, train2_y)
predictions2 = model2.predict(test_x)

accuracy2 = accuracy_score(test_y, predictions2) * 100
print("The accuracy was %.2f%%" % accuracy2)

# show head columns
# print(x.head())

# show elements quantity
# print(train2_x.shape)
# print(test2_x.shape)
