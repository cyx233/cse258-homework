import json
import gzip
import math
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy
from sklearn import linear_model, metrics
import random
import statistics

random.seed(0)


def assertFloat(x):
    assert type(float(x)) == float


def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float] * N


answers = {}
z = gzip.open("train.json.gz")
dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)
z.close()


### Question 1
def MSE(y, ypred):
    return metrics.mean_squared_error(y, ypred)


def MAE(y, ypred):
    return metrics.mean_absolute_error(y, ypred)

def impulse_func(cond):
    return 1 if cond else 0


reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u, i = d["userID"], d["gameID"]
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)

for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x["date"])

for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x["date"])


def feat1(d):
    return (1, d["hours"])


X = [feat1(d) for d in dataset]
y = [len(d["text"]) for d in dataset]
mod = linear_model.LinearRegression()
mod.fit(X, y)
predictions = mod.predict(X)
theta_1 = mod.coef_[1]
mse_q1 = MSE(y, predictions)

answers["Q1"] = [theta_1, mse_q1]
assertFloatList(answers["Q1"], 2)
### Question 2

median_hours = numpy.median([d["hours"] for d in dataset])


def feat2(d):
    return (
        1,
        d["hours"],
        d["hours_transformed"],
        numpy.square(d["hours"]),
        impulse_func(d["hours"] > median_hours),
    )


X = [feat2(d) for d in dataset]
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X, y)
predictions = mod.predict(X)
mse_q2 = MSE(y, predictions)

answers["Q2"] = mse_q2
assertFloat(answers["Q2"])


### Question 3
def feat3(d):
    h = d["hours"]
    return (
        1,
        impulse_func(h > 1),
        impulse_func(h > 5),
        impulse_func(h > 10),
        impulse_func(h > 100),
        impulse_func(h > 1000),
    )


X = [feat3(d) for d in dataset]
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X, y)
predictions = mod.predict(X)
mse_q3 = MSE(y, predictions)

answers["Q3"] = mse_q3
assertFloat(answers["Q3"])
### Question 4
def feat4(d):
    return (1, len(d['text']))

X = [feat4(d) for d in dataset]
y = [d['hours'] for d in dataset]
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)
mse = MSE(y, predictions)
mae = MAE(y, predictions)

answers['Q4'] = [mse, mae, "explain which is better"]
assertFloatList(answers['Q4'][:2], 2)
### Question 5

y_trans = [d['hours_transformed'] for d in dataset]
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)
mse_trans = MSE(y_trans, predictions_trans) # MSE using the transformed variable

predictions_untrans = 2**(predictions_trans-1) # Undoing the transformation
mse_untrans = MSE(y, predictions_untrans)
answers['Q5'] = [mse_trans, mse_untrans]
assertFloatList(answers['Q5'], 2)
### Question 6
def feat6(d):
    feat = numpy.zeros(100)
    if d['hours'] >= 99:
        feat[99] = 1
    else:
        feat[int(d['hours'])] = 1
    return feat

X = [feat6(d) for d in dataset]
y = [len(d['text']) for d in dataset]
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]
models = {}
mses = {}
bestC = None

for c in [1, 10, 100, 1000, 10000]:
    mod = linear_model.Ridge(c)
    mod.fit(Xtrain, ytrain)
    predictions = mod.predict(Xvalid)
    mses[c] = MSE(yvalid, predictions)
    models[c] = mod
    if not bestC or mses[c] < mses[bestC]:
        bestC = c

predictions_test = models[bestC].predict(Xtest)
mse_valid = mses[bestC]
mse_test = MSE(ytest, predictions_test)
answers['Q6'] = [bestC, mse_valid, mse_test]
assertFloatList(answers['Q6'], 3)
### Question 7
times = [d['hours_transformed'] for d in dataset]
median = statistics.median(times)

notPlayed = [impulse_func(d['hours'] < 1) for d in dataset]
nNotPlayed = sum(notPlayed)
answers['Q7'] = [median, nNotPlayed]
assertFloatList(answers['Q7'], 2)
### Question 8
def feat8(d):
    return (len(d['text']), )

X = [feat8(d) for d in dataset]
y = numpy.array([d['hours_transformed'] > median for d in dataset])
mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X,y)
predictions = mod.predict(X) # Binary vector of predictions
def rates(predictions, y):
    return metrics.confusion_matrix(y, predictions).ravel()
TN, FP, FN, TP = rates(predictions, y)
BER = 1 - metrics.balanced_accuracy_score(y, predictions)
answers['Q8'] = [TP, TN, FP, FN, BER]
assertFloatList(answers['Q8'], 5)
### Question 9
precision = TP / (TP + FP)
recall = TP / (TP + FN)

precs = []
recs = []
prob_positive_class = mod.predict_proba(X)[:, 1]

sorted_indices = prob_positive_class.argsort()[::-1]
sorted_probs = prob_positive_class[sorted_indices]

for i in [5, 10, 100, 1000]:
    threshold_score = sorted_probs[i-1] 
    mask = prob_positive_class >= threshold_score
    precs.append(metrics.precision_score(y[mask], predictions[mask]))

answers['Q9'] = precs
assertFloatList(answers['Q9'], 4)
### Question 10
X = [feat4(d) for d in dataset]
y_trans = [d['hours_transformed'] for d in dataset]
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)
threshold = 3.7
predictions_thresh = predictions_trans > threshold # Using a fixed threshold to make predictions
TN, FP, FN, TP = rates(y, predictions_thresh)
BER = 1 - metrics.balanced_accuracy_score(y, predictions_thresh)
answers['Q10'] = [threshold, BER]
assertFloatList(answers['Q10'], 2)
### Question 11
dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]
userMedian = defaultdict(list)
itemMedian = defaultdict(list)

# Compute medians on training data
for d in dataTrain:
    userMedian[d['userID']].append(d['hours'])
    itemMedian[d['gameID']].append(d['hours'])

for u in userMedian:
    userMedian[u] = numpy.median(userMedian[u])

for g in itemMedian:
    itemMedian[g] = numpy.median(itemMedian[g])

answers['Q11'] = [itemMedian['g35322304'], userMedian['u55351001']]
assertFloatList(answers['Q11'], 2)
### Question 12
global_median = numpy.median([d["hours"] for d in dataset])
def f12(u,i):
    # Function returns a single value (0 or 1)
    if i in itemMedian:
        if itemMedian[i] > global_median:
            return 1
    elif u in userMedian and userMedian[u] > global_median:
        return 1
    return 0
preds = [f12(d['userID'], d['gameID']) for d in dataTest]
y = [impulse_func(d['hours'] > global_median) for d in dataTest]
accuracy = metrics.accuracy_score(y, preds)
answers['Q12'] = accuracy
assertFloat(answers['Q12'])
### Question 13
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}

for d in dataset:
    user,item = d['userID'], d['gameID']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)

def Jaccard(s1, s2):
    num = len(s1 & s2)
    den = len(s1 | s2)
    return num / den if den > 0 else 0

def mostSimilar(i, func, N):
    ans = []
    for i2 in set(usersPerItem.keys())-{i}:
        s1 = usersPerItem[i]
        s2 = usersPerItem[i2]
        ans.append((func(s1, s2), i2))
    ans.sort(reverse=True)
    return ans[:N]

ms = mostSimilar(dataset[0]['gameID'], Jaccard, 10)
answers['Q13'] = [ms[0][0], ms[-1][0]]
assertFloatList(answers['Q13'], 2)
### Question 14
def mostSimilar14(i, func, N):
    ans = []
    for i2 in set(usersPerItem.keys())-{i}:
        if i != i2:
            ans.append((func(i, i2), i2))
    # Sort based on similarity
    ans.sort(reverse=True)  # Now sorting based on the similarity score
    return ans[:N]

ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = 1 if d['hours'] > global_median else -1 # Set the label based on a rule
    ratingDict[(u,i)] = lab
def Cosine(i1, i2):
    # Between two items
    vec1 = numpy.array([ratingDict.get((u, i1), 0) for u in itemsPerUser])
    vec2 = numpy.array([ratingDict.get((u, i2), 0) for u in itemsPerUser])
    
    dot_product = numpy.dot(vec1, vec2)
    norm1 = numpy.linalg.norm(vec1)
    norm2 = numpy.linalg.norm(vec2)
    similarity = dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0
    return similarity

ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)
answers['Q14'] = [ms[0][0], ms[-1][0]]
assertFloatList(answers['Q14'], 2)
### Question 15
ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = d['hours_transformed'] # Set the label based on a rule
    ratingDict[(u,i)] = lab
ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)
answers['Q15'] = [ms[0][0], ms[-1][0]]
assertFloatList(answers['Q15'], 2)
f = open("answers_midterm.txt", "w")
f.write(str(answers) + "\n")
f.close()
