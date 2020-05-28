import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

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

# divide data (75 for training and 24 for test)
training_x = x[:75]
training_y = y[:75]
test_x = x[75:]
test_y = y[75:]

print("We trained with %d elements and will test with %d elements" % (len(training_x), len(test_x)))

# training the model with people interactions (training_x) and if they bought a product (training_y)
model = LinearSVC()
model.fit(training_x, training_y)

# predict if they bought given the people interaction (text_x data)
predictions = model.predict(test_x)

# percent of correct predictions according with real case (test_y)
accuracy = accuracy_score(test_y, predictions) * 100
print("The accuracy was %d.2f%%" % accuracy)


