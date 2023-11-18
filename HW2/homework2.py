from math import exp
import random
from re import A
import numpy as np
from sklearn import linear_model, metrics
from matplotlib import pyplot as plt
from collections import defaultdict
import gzip


def assertFloat(x):
    assert type(float(x)) == float


def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float] * N


answers = {}


def parseData(fname):
    for l in open(fname):
        yield eval(l)


data = list(parseData("beer_50000.json"))
random.seed(0)
random.shuffle(data)
dataTrain = data[:25000]
dataValid = data[25000:37500]
dataTest = data[37500:]
yTrain = [d["beer/ABV"] > 7 for d in dataTrain]
yValid = [d["beer/ABV"] > 7 for d in dataValid]
yTest = [d["beer/ABV"] > 7 for d in dataTest]

categoryCounts = defaultdict(int)
for d in data:
    categoryCounts[d["beer/style"]] += 1
categories = [c for c in categoryCounts if categoryCounts[c] > 1000]
catID = dict(zip(list(categories), range(len(categories))))

max_len = max([len(d["review/text"]) for d in data])


def feat(d, includeCat=True, includeReview=True, includeLength=True):
    # In my solution, I wrote a reusable function that takes parameters to generate features for each question
    # Feel free to keep or discard

    features = []
    cat_one_hot = np.zeros(len(catID) + 1, dtype=np.float64)
    if d["beer/style"] in catID:
        cat_one_hot[catID[d["beer/style"]]] = 1
    rating = np.array(
        [
            d["review/appearance"],
            d["review/palate"],
            d["review/taste"],
            d["review/overall"],
            d["review/aroma"],
        ],
        dtype=np.float64,
    )
    review_len = np.array([len(d["review/text"]) / max_len], dtype=np.float64)

    if includeCat:
        features.append(cat_one_hot)
    if includeReview:
        features.append(rating)
    if includeLength:
        features.append(review_len)
    return np.concatenate(features)


def pipeline(reg, includeCat=True, includeReview=True, includeLength=True):
    # Extract features from training data
    XTrain = np.stack(
        [feat(d, includeCat, includeReview, includeLength) for d in dataTrain]
    )

    # Train logistic regression model
    mod = linear_model.LogisticRegression(C=reg, max_iter=1000)
    mod.fit(XTrain, yTrain)

    # Extract features from validation and test data
    XValid = [feat(d, includeCat, includeReview, includeLength) for d in dataValid]
    XTest = [feat(d, includeCat, includeReview, includeLength) for d in dataTest]

    # Make predictions on validation and test data
    validPreds = mod.predict(XValid)
    testPreds = mod.predict(XTest)

    # Calculate BER for validation and test data
    validBER = 1 - metrics.balanced_accuracy_score(yValid, validPreds)
    testBER = 1 - metrics.balanced_accuracy_score(yTest, testPreds)

    return mod, validBER, testBER


### Question 1
mod, validBER, testBER = pipeline(10, True, False, False)
answers["Q1"] = [validBER, testBER]
assertFloatList(answers["Q1"], 2)
### Question 2
mod, validBER, testBER = pipeline(10, True, True, True)
answers["Q2"] = [validBER, testBER]
assertFloatList(answers["Q2"], 2)
### Question 3
best_validBER = 1
bestC = 0
for c in [0.001, 0.01, 0.1, 1, 10]:
    mod, validBER, testBER = pipeline(c, True, True, True)
    if validBER < best_validBER:
        bestC = c
mod, validBER, testBER = pipeline(bestC, True, True, True)
answers["Q3"] = [bestC, validBER, testBER]
assertFloatList(answers["Q3"], 3)
### Question 4
mod, validBER, testBER_noCat = pipeline(bestC, False, True, True)
mod, validBER, testBER_noReview = pipeline(bestC, True, False, True)
mod, validBER, testBER_noLength = pipeline(bestC, True, True, False)
answers["Q4"] = [testBER_noCat, testBER_noReview, testBER_noLength]
assertFloatList(answers["Q4"], 3)
### Question 5
path = "amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz"
f = gzip.open(path, "rt", encoding="utf8")

header = f.readline()
header = header.strip().split("\t")
print(header)
dataset = []
pairsSeen = set()

for line in f:
    fields = line.strip().split("\t")
    d = dict(zip(header, fields))
    ui = (d["customer_id"], d["product_id"])
    if ui in pairsSeen:
        print("Skipping duplicate user/item:", ui)
        continue
    pairsSeen.add(ui)
    d["star_rating"] = int(d["star_rating"])
    d["helpful_votes"] = int(d["helpful_votes"])
    d["total_votes"] = int(d["total_votes"])
    dataset.append(d)
dataTrain = dataset[: int(len(dataset) * 0.9)]
dataTest = dataset[int(len(dataset) * 0.9) :]
# Feel free to keep or discard

usersPerItem = defaultdict(set)  # Maps an item to the users who rated it
itemsPerUser = defaultdict(set)  # Maps a user to the items that they rated

for d in dataset:
    user, item, rating = d["customer_id"], d["product_id"], d["star_rating"]
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)


def Jaccard(s1, s2):
    num = len(s1.intersection(s2))
    den = len(s1.union(s2))
    return num / den if den > 0 else 0


def mostSimilar(i, N):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i:
            continue
        sim = Jaccard(users, usersPerItem[i2])
        similarities.append((sim, i2))
    similarities.sort(reverse=True)
    return similarities[:N]


query = "B00KCHRKD6"
ms = mostSimilar(query, 10)
answers["Q5"] = ms
assertFloatList([m[0] for m in ms], 10)

### Question 6
usersPerItem = defaultdict(set)  # Maps an item to the users who rated it
itemsPerUser = defaultdict(set)  # Maps a user to the items that they rated
itemNames = {}
ratingDict = {}  # To retrieve a rating for a specific user/item pair
reviewsPerUser = defaultdict(list)
dateDict = {}  # To retrieve a date for a specific user/item pair

for d in dataTrain:
    user, item, rating, date = (
        d["customer_id"],
        d["product_id"],
        d["star_rating"],
        d["review_date"],
    )
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    ratingDict[(user, item)] = rating
    dateDict[(user, item)] = date
    itemNames[item] = d["product_title"]

userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    userAverages[u] = sum([ratingDict[(u, i)] for i in itemsPerUser[u]]) / len(
        itemsPerUser[u]
    )

for i in usersPerItem:
    itemAverages[i] = sum([ratingDict[(u, i)] for u in usersPerItem[i]]) / len(
        usersPerItem[i]
    )

ratingMean = sum([ratingDict[(u, i)] for u, i in ratingDict]) / len(ratingDict)


def MSE(y, ypred):
    return metrics.mean_squared_error(y, ypred)


def predictRating(user, item):
    if item not in usersPerItem or user not in itemsPerUser:
        return ratingMean
    simSum = 0
    weightedSum = 0
    for item2 in itemsPerUser[user]:
        if item2 == item:
            continue
        sim = Jaccard(usersPerItem[item], usersPerItem[item2])
        if sim > 0:
            weightedSum += (ratingDict[(user, item2)] - itemAverages[item2]) * sim
            simSum += sim
    if simSum == 0:
        return itemAverages.get(item, ratingMean)
    return itemAverages[item] + weightedSum / simSum


simPredictions = [predictRating(d["customer_id"], d["product_id"]) for d in dataTest]
labels = [d["star_rating"] for d in dataTest]
answers["Q6"] = MSE(simPredictions, labels)
assertFloat(answers["Q6"])

### Question 7

from datetime import datetime


def decay_func(date_i, date_j, decay_constant):
    date_format = "%Y-%m-%d"
    date_i = datetime.strptime(date_i, date_format)
    date_j = datetime.strptime(date_j, date_format)
    date_difference = abs((date_j - date_i).days)
    return exp(-decay_constant * date_difference)


def predictRatingWithDecay(user, item, date):
    if item not in usersPerItem or user not in itemsPerUser:
        return ratingMean
    simSum = 0
    weightedSum = 0
    for item2 in itemsPerUser[user]:
        if item2 == item:
            continue
        decay = decay_func(date, dateDict[(user, item2)], 1)
        sim = Jaccard(usersPerItem[item], usersPerItem[item2])
        if sim > 0:
            weightedSum += (
                (ratingDict[(user, item2)] - itemAverages[item2]) * sim * decay
            )
            simSum += sim * decay
    if simSum == 0:
        return itemAverages.get(item, ratingMean)
    return itemAverages[item] + weightedSum / simSum


simPredictions = [
    predictRatingWithDecay(d["customer_id"], d["product_id"], d["review_date"])
    for d in dataTest
]
labels = [d["star_rating"] for d in dataTest]
itsMSE = MSE(simPredictions, labels)
answers["Q7"] = ["In my homework, f(t_ui, t_uj)=e^|t_ui-t_uj|", itsMSE]
assertFloat(answers["Q7"][1])

f = open("answers_hw2.txt", "w")
f.write(str(answers) + "\n")
f.close()
