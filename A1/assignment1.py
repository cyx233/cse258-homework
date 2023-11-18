from collections import defaultdict
import math
from tqdm import tqdm
import gzip
import random
import heapq

import numpy
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics


random.seed(42)

"""
userID
gameID
hours
hours_transformed
early_access
text
date
"""


def readJSON(path):
    f = gzip.open(path, "rt")
    f.readline()
    for l in f:
        d = eval(l)
        yield d["userID"], d["gameID"], d["hours_transformed"]


allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)
random.shuffle(allHours)
hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

user_game = defaultdict(set)
game_user = defaultdict(set)
user_hours = defaultdict(dict)
game_hours = defaultdict(dict)

for i in hoursTrain:
    user_game[i[0]].add(i[1])
    game_user[i[1]].add(i[0])
    user_hours[i[0]][i[1]] = i[2]
    game_hours[i[1]][i[0]] = i[2]


allGame = set(game_user.keys())
negHoursTrain = [
    (i[0], random.sample(list(allGame - user_game[i[0]]), 1)[0]) for i in hoursTrain
]

negHoursValid = [
    (i[0], random.sample(list(allGame - user_game[i[0]]), 1)[0]) for i in hoursValid
]


### Time-played
### baseline    4.988273760366987
### now         3.0629636914492
### grade       1.5/2
def iterate(lamb, alpha, betaU, betaI):
    new_betaU = {
        u: sum([user_hours[u][g] - (alpha + betaI[g]) for g in user_game[u]]) / (lamb + len(user_game[u]))
        for u in betaU
    }
    new_betaI = {
        g: sum([user_hours[u][g] - (alpha + betaU[u]) for u in game_user[g]]) / (lamb + len(game_user[g]))
        for g in betaI
    }
    a = sum(
        [(user_hours[u][g] - (new_betaU[u] + new_betaI[g])) for u, g, _ in hoursTrain]
    ) / len(hoursTrain)
    return a, new_betaU, new_betaI


alpha = numpy.median([i[2] for i in hoursTrain])
betaU = {u: 0 for u in user_game}
betaI = {g: 0 for g in game_user}
for i in range(10):
    alpha, betaU, betaI = iterate(5, alpha, betaU, betaI)

preds = []
labels = []
for u, g, h in hoursValid:
    preds.append(
        alpha
        + betaU.get(u, numpy.average([i for i in betaU.values()]))
        + betaI.get(g, numpy.average([i for i in betaI.values()]))
    )
    labels.append(h)
print(metrics.mean_squared_error(labels, preds))

with open("predictions_Hours.csv", "w") as f:
    for l in open("pairs_Hours.csv"):
        if l.startswith("userID"):
            f.write(l)
            continue
        u, g = l.strip().split(",")

        # Logic
        pred = (
            alpha
            + betaU.get(u, numpy.average([i for i in betaU.values()]))
            + betaI.get(g, numpy.average([i for i in betaI.values()]))
        )

        _ = f.write(u + "," + g + "," + str(pred) + "\n")


### Would-play
### baseline    0.6806
### now         0.6929
## grade        1/2
medianUser = numpy.median([len(i) for i in user_game.values()])
medianGame = numpy.median([len(i) for i in game_user.values()])


def jaccard_simularity(s1, s2):
    num = len(s1 & s2)
    den = len(s1 | s2)
    return num / den if den > 0 else 0


def topN_jaccard(u, g, N):
    ans = [0 for _ in range(N)]
    for g2 in user_game.get(u, set()):
        if g != g2:
            heapq.heappushpop(
                ans,
                jaccard_simularity(game_user.get(g, set()), game_user.get(g2, set())),
            )
    ans.sort(reverse=True)
    return ans[:N]


def feat(d):
    u, g = d[0], d[1]
    user_pop = math.log(medianUser)
    game_pop = math.log(medianGame)
    if u in user_game:
        user_pop = math.log(len(user_game[u]))
    if g in game_user:
        game_pop = math.log(len(game_user[g]))
    return numpy.array([user_pop, game_pop] + topN_jaccard(u, g, 1))


clf = linear_model.LogisticRegression(class_weight="balanced")
X = [feat(d) for d in tqdm(hoursTrain + negHoursTrain)]
Y = [1 for _ in range(len(hoursTrain))] + [0 for _ in range(len(negHoursTrain))]
clf.fit(X, Y)

validX = [feat(d) for d in tqdm(hoursValid + negHoursValid)]
validY = [1 for _ in range(len(hoursValid))] + [0 for _ in range(len(negHoursValid))]
pred = clf.predict(validX)
print(metrics.accuracy_score(validY, pred))

with open("predictions_Played.csv", "w") as f:
    testX = []
    users = []
    games = []
    for l in open("pairs_Played.csv"):
        if l.startswith("userID"):
            f.write(l)
            continue
        u, g = l.strip().split(",")
        users.append(u)
        games.append(g)
        testX.append(feat([u, g]))
    # Logic
    for u, g, pred in zip(users, games, clf.predict(testX)):
        _ = f.write(u + "," + g + "," + str(pred) + "\n")
