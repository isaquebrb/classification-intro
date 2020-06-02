from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# predict between woman and man

# Features (1 yes, 0 no)
# long hair
# thick voice
# muscular arms
woman1 = [1, 1, 0]
woman2 = [0, 1, 1]
woman3 = [1, 1, 0]
woman4 = [1, 0, 0]
woman5 = [1, 1, 0]

man1 = [0, 0, 1]
man2 = [0, 1, 0]
man3 = [1, 0, 1]
man4 = [0, 0, 0]
man5 = [1, 0, 1]

# 0 -> woman, 1 -> man
train_x = [woman1, woman2, woman3, woman4, woman5, man1, man2, man3, man4, man5]
train_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# train the "machine" with given data (x = persons, y = correct classification)
model = LinearSVC()
model.fit(train_x, train_y)

# create random persons
mystery1 = [1, 1, 1]
mystery2 = [1, 1, 0]
mystery3 = [0, 1, 1]

# predict random persons classification
test_x = [mystery1, mystery2, mystery3]
prediction = model.predict(test_x)

# correct classification
test_y = [0, 1, 1]

# calculate correct rate %
corrects = (prediction == test_y).sum()
total = len(test_x)
correct_rate = corrects / total
print("Correct rate: ", correct_rate * 100)

# calculate correct rate % with metrics
correct_rate = accuracy_score(test_y, prediction)
print("Correct rate: ", correct_rate * 100)
