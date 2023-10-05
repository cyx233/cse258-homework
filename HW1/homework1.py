import json
from collections import defaultdict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy
import random
import gzip
import dateutil.parser
import math

answers = {}


def assertFloat(x):
    assert type(float(x)) == float


def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float] * N


### Question 1
f = gzip.open("fantasy_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

max_len = max(map(lambda x: len(x["review_text"]), dataset))
def feature(datum):
    return [1, len(datum["review_text"])/max_len]


X = list(map(feature, dataset))
Y = list(map(lambda x: x['rating'], dataset))

# Train the linear regression model
model = linear_model.LinearRegression()
model.fit(X, Y)
# Get theta0 and theta1
theta = model.intercept_, model.coef_[1]
# Predict the Y values
Y_pred = model.predict(X)
# Calculate the Mean Squared Error
MSE = mean_squared_error(Y, Y_pred)

answers['Q1'] = [theta[0], theta[1], MSE]
assertFloatList(answers['Q1'], 3)

### Question 2
for d in dataset:
    t = dateutil.parser.parse(d['date_added'])
    d['parsed_date'] = t

def feature(datum):
    weekday_feature = [0 for i in range(7)]
    weekday_feature[datum['parsed_date'].weekday()] = 1
    month_feature = [0 for i in range(12)]
    month_feature[datum['parsed_date'].month-1] = 1
    return weekday_feature + month_feature


X = list(map(feature, dataset))
Y = list(map(lambda x: x['rating'], dataset))

answers['Q2'] = [X[0], X[1]]
assertFloatList(answers['Q2'][0], 19)
assertFloatList(answers['Q2'][1], 19)


### Question 3
def feature3(datum):
    return [1, len(datum["review_text"])/max_len, datum['parsed_date'].weekday(), datum['parsed_date'].month]


X3 = list(map(feature, dataset))
Y3 = list(map(lambda x: x['rating'], dataset))

answers['Q3'] = [mse2, mse3]
assertFloatList(answers['Q3'], 2)

### Question 4
random.seed(0)
random.shuffle(dataset)

X2 = [feature(d) for d in dataset]
X3 = [feature3(d) for d in dataset]
Y = [d['rating'] for d in dataset]

train2, test2 = X2[:len(X2) // 2], X2[len(X2) // 2:]
train3, test3 = X3[:len(X3) // 2], X3[len(X3) // 2:]
trainY, testY = Y[:len(Y) // 2], Y[len(Y) // 2:]

answers['Q4'] = [test_mse2, test_mse3]
assertFloatList(answers['Q4'], 2)

### Question 5
f = open("beer_50000.json")
dataset = []
for l in f:
    dataset.append(eval(l))

# X =
# Y =

# TP =
# TN =
# FP =
# FN =
# BER =

answers['Q5'] = [TP, TN, FP, FN, BER]
assertFloatList(answers['Q5'], 5)

### Question 6
precs = []
for k in [1, 100, 1000, 10000]:
    pass
answers['Q6'] = precs
assertFloatList(answers['Q6'], 4)

### Question 7
its_test_BER = 1000
answers['Q7'] = ["Description of my solution", its_test_BER]
f = open("answers_hw1.txt", 'w')
f.write(str(answers) + '\n')
f.close()
