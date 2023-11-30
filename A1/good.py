from collections import defaultdict
from tqdm import tqdm
import gzip
import random
import heapq

import numpy
import matplotlib.pyplot as plt
from sklearn import metrics


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

for i in allHours:
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
### now         3.054113872717232
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
    alpha, betaU, betaI = iterate(4.65, alpha, betaU, betaI)

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
### now         0.7332

user_game = defaultdict(set)
game_user = defaultdict(set)
user_hours = defaultdict(dict)
# nodes
allGame = set()
allUser = set()
for u, g, _ in allHours:
    allGame.add(g)
    allUser.add(u)

# edges
popGame = defaultdict(int)
for u, g, h in hoursTrain:
    user_game[u].add(g)
    game_user[g].add(u)
    user_hours[u][g] = h
    popGame[g] += 1

all_user_game = defaultdict(set)
for u, g, _ in allHours:
    user_game[u].add(g)

negHoursValid = [
    (i[0], random.sample(list(allGame - all_user_game[i[0]]), 1)[0], i[2])
    for i in hoursValid
]

averageUserVisit = numpy.average([len(i) for i in user_game.values()])
averageGameVisited = numpy.average([len(i) for i in game_user.values()])


def jaccard_simularity(s1, s2):
    num = len(s1 & s2)
    den = len(s1 | s2)
    return num / den if den > 0 else 0


def kNN(k, users, user_game):
    k_neighbor = defaultdict(set)
    for u1 in tqdm(users):
        neighbors = []
        for u2 in users:
            similarity = jaccard_simularity(user_game[u1], user_game[u2])
            if len(neighbors) < k:
                heapq.heappush(neighbors, (similarity, u2))
            else:
                heapq.heappushpop(neighbors, (similarity, u2))
        k_neighbor[u1] = set([i[1] for i in neighbors])
    return k_neighbor


X = defaultdict(dict)
users_k_neighbor = kNN(560, allUser, user_game)
for u in tqdm(allUser):
    prefer_games = defaultdict(int)
    for neighbor in users_k_neighbor[u]:
        for g in user_game[neighbor]:
            prefer_games[g] += 1 / len(user_game[neighbor])
    for g in prefer_games:
        X[u][g] = prefer_games[g]

sorted_X = {}
for u in allUser:
    sorted_X[u] = sorted(X[u].values(), reverse=True)

best_acc = 0
labels = [1 for _ in range(len(hoursValid))] + [0 for _ in range(len(negHoursValid))]

def would_play(data):
    user_game_score = defaultdict(list)
    for u, g, _ in data:
        score = X[u].get(g, 0)
        user_game_score[u].append(score)

    preds = []
    for u, g, _ in data:
        score = X[u].get(g, 0)
        preds.append(1 if score > numpy.median(user_game_score[u]) else 0)
    return preds


preds = would_play(hoursValid + negHoursValid)
acc = metrics.accuracy_score(labels, preds)
print(f"{acc}, {numpy.mean(preds)}")

with open("predictions_Played.csv", "w") as f:
    data = []
    for l in open("pairs_Played.csv"):
        if l.startswith("userID"):
            f.write(l)
            continue
        u, g = l.strip().split(",")
        data.append((u, g, 0))
    preds = would_play(data)
    for d, pred in zip(data, preds):
        u, g, h = d
        _ = f.write(u + "," + g + "," + str(pred) + "\n")
print(numpy.mean(preds))