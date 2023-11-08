import gzip
from collections import defaultdict
import math
from matplotlib import pyplot as plt
import scipy.optimize
from sklearn import metrics, svm
import numpy
import string
import random
import string
from sklearn import linear_model

random.seed(0)


def assertFloat(x):
    assert type(float(x)) == float


def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float] * N


def readGz(path):
    for l in gzip.open(path, "rt"):
        yield eval(l)


def readJSON(path):
    f = gzip.open(path, "rt")
    f.readline()
    for l in f:
        d = eval(l)
        u = d["userID"]
        g = d["gameID"]
        yield u, g, d


answers = {}
# Some data structures that will be useful
allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)
hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]
##################################################
# Play prediction                                #
##################################################

### Question 1
# Evaluate baseline strategy
gameCount = defaultdict(int)
allGame = set()
for i in allHours:
    gameCount[i[1]] += 1
    allGame.add(i[1])
totalPlayed = len(allHours)
mostPopular = sorted([(gameCount[x], x) for x in gameCount], reverse=True)


def baseline(dataValid, threshold):
    count = 0
    return1 = set()
    limit = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        limit = ic
        if count > totalPlayed * threshold:
            break
    pred = numpy.zeros(len(dataValid))
    for i, v in enumerate(dataValid):
        if v[1] in return1:
            pred[i] = 1
    return pred


user_game = defaultdict(set)

for i in allHours:
    user_game[i[0]].add(i[1])

negHoursTrain = [
    (i[0], random.sample(list(allGame - user_game[i[0]]), 1)[0], i[2])
    for i in hoursTrain
]
negHoursValid = [
    (i[0], random.sample(list(allGame - user_game[i[0]]), 1)[0], i[2])
    for i in hoursValid
]

train_labels = [1] * len(hoursTrain) + [0] * len(negHoursTrain)
valid_labels = [1] * len(hoursValid) + [0] * len(negHoursValid)

pred = baseline(hoursValid + negHoursValid, 0.5)
answers["Q1"] = metrics.accuracy_score(valid_labels, pred)
### Question 2
# Improved strategy
# Evaluate baseline strategy
threshold = 0.7
pred = baseline(hoursValid + negHoursValid, threshold)
answers["Q2"] = [threshold, metrics.accuracy_score(valid_labels, pred)]
assertFloatList(answers["Q2"], 2)

### Question 3/4
user_game = defaultdict(set)
game_user = defaultdict(set)
for i in hoursTrain:
    user_game[i[0]].add(i[1])
    game_user[i[1]].add(i[0])


def jaccard_simularity(g1, g2):
    u1 = game_user[g1]
    u2 = game_user[g2]
    num = len(u1 & u2)
    den = len(u1 | u2)
    return num / den if den > 0 else 0


def jaccard_feat(u, g):
    ans = 0
    for g2 in user_game[u]:
        if g != g2:
            ans = max(ans, jaccard_simularity(g, g2))
    return ans


def jaccard_classifier(dataValid, threshold):
    pred = numpy.zeros(len(dataValid))
    feats = []
    for i, v in enumerate(dataValid):
        feat = jaccard_feat(v[0], v[1])
        feats.append(feat)
        if feat > threshold:
            pred[i] = 1
    return pred


def combination_calssifier(dataValid):
    count = 0
    return1 = set()
    limit = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        limit = ic
        if count > totalPlayed * 0.7:
            break
    pred = numpy.zeros(len(dataValid))
    for i, v in enumerate(dataValid):
        feat = jaccard_feat(v[0], v[1])
        if feat > 0.06:
            pred[i] = 1
        elif v[1] in return1 and feat > 0.015:
            pred[i] = 1
    return pred


pred = jaccard_classifier(hoursValid + negHoursValid, 0.035)
answers["Q3"] = metrics.accuracy_score(valid_labels, pred)

pred = combination_calssifier(hoursValid + negHoursValid)
answers["Q4"] = metrics.accuracy_score(valid_labels, pred)

assertFloat(answers["Q3"])
assertFloat(answers["Q4"])

answers["Q5"] = "I confirm that I have uploaded an assignment submission to gradescope"
##################################################
# Hours played prediction                        #
##################################################
labels = [d["hours_transformed"] for u, g, d in hoursValid]


### Question 6
betaU = {}
betaI = {}
r_ui = {}
globalAverage = 0
for u, g, d in hoursTrain:
    betaU[u] = 0
    betaI[g] = 0
    r_ui[(u, g)] = d["hours_transformed"]
    globalAverage += d["hours_transformed"]
globalAverage /= len(hoursTrain)


alpha = globalAverage  # Could initialize anywhere, this is a guess


def iterate(lamb, alpha, betaU, betaI):
    temp = 0
    for u, g, d in hoursTrain:
        temp += r_ui[(u, g)] - (betaU[u] + betaI[g])
    alpha = temp / len(hoursTrain)

    for u in betaU:
        temp = 0
        for g in user_game[u]:
            temp += r_ui[(u, g)] - (alpha + betaI[g])
        betaU[u] = temp / (lamb + len(user_game[u]))

    for g in betaI:
        temp = 0
        for u in game_user[g]:
            temp += r_ui[(u, g)] - (alpha + betaU[u])
        betaI[g] = temp / (lamb + len(game_user[g]))
    return alpha


for i in range(10):
    alpha = iterate(1, alpha, betaU, betaI)

preds = []
labels = []
for u, g, d in hoursValid:
    preds.append(alpha + betaU[u] + betaI[g])
    labels.append(d["hours_transformed"])

validMSE = metrics.mean_squared_error(labels, preds)
plt.scatter(list(range(len(labels))), labels, c="r", s=1)
plt.scatter(list(range(len(preds))), preds, c="b", s=1)
plt.savefig("Q6")

answers["Q6"] = validMSE
assertFloat(answers["Q6"])
### Question 7
betaUs = [(betaU[u], u) for u in betaU]
betaIs = [(betaI[i], i) for i in betaI]
betaUs.sort()
betaIs.sort()

print("Maximum betaU = " + str(betaUs[-1][1]) + " (" + str(betaUs[-1][0]) + ")")
print("Maximum betaI = " + str(betaIs[-1][1]) + " (" + str(betaIs[-1][0]) + ")")
print("Minimum betaU = " + str(betaUs[0][1]) + " (" + str(betaUs[0][0]) + ")")
print("Minimum betaI = " + str(betaIs[0][1]) + " (" + str(betaIs[0][0]) + ")")
answers["Q7"] = [betaUs[-1][0], betaUs[0][0], betaIs[-1][0], betaIs[0][0]]
answers["Q7"]
assertFloatList(answers["Q7"], 4)
### Question 8
validMSE = 100
best_lambda = 5
for lamb in [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10]:
    for i in range(10):
        alpha = iterate(lamb, alpha, betaU, betaI)
    preds = []
    for u, g, d in hoursValid:
        preds.append(alpha + betaU[u] + betaI[g])
    mse = metrics.mean_squared_error(labels, preds)
    if mse < validMSE:
        validMSE = mse
        best_lambda = lamb

answers["Q8"] = (best_lambda, validMSE)
assertFloatList(answers["Q8"], 2)

f = open("answers_hw3.txt", "w")
f.write(str(answers) + "\n")
f.close()
