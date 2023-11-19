from collections import defaultdict
import copy
import math
from unittest.util import sorted_list_difference
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


### Time-played
### baseline    4.988273760366987
### now         3.054113872717232
### grade       1.5/2

# nodes
allGame = set()
allUser = set()
for u, g, _ in allHours:
    allGame.add(g)
    allUser.add(u)

user_game = defaultdict(set)
game_user = defaultdict(set)
# edges
for u, g, h in hoursTrain:
    user_game[u].add((g, h))
    game_user[g].add((u, h))

all_user_game = defaultdict(set)
for u, g, _ in allHours:
    user_game[u].add((g, h))

negHoursValid = [
    (i[0], random.sample(list(allGame - all_user_game[i[0]]), 1)[0], i[2])
    for i in hoursValid
]


def update_alpha_beta(lamb, alpha, betaU, betaG, gammaU, gammaG):
    new_betaU = {
        u: sum(
            [
                h - alpha - betaG[g] - gammaU[u].dot(gammaG[g].transpose())[0][0]
                for g, h in user_game[u]
            ]
        )
        / (lamb + len(user_game[u]))
        for u in user_game
    }
    new_betaI = {
        g: sum(
            [
                h - alpha - betaU[u] - gammaU[u].dot(gammaG[g].transpose())[0][0]
                for u, h in game_user[g]
            ]
        )
        / (lamb + len(game_user[g]))
        for g in game_user
    }
    a = sum(
        [
            (
                h
                - new_betaU[u]
                - new_betaI[g]
                - gammaU[u].dot(gammaG[g].transpose())[0][0]
            )
            for u, g, h in hoursTrain
        ]
    ) / len(hoursTrain)
    return a, new_betaU, new_betaI


def update_gamma_u(alpha, betaU, betaG, gammaU, gammaG, rate):
    gammaU = copy.deepcopy(gammaU)
    total = len(hoursTrain)
    # Update gammaU
    for u in user_game:
        sum_for_gammaU = 0
        update_flag = bool(random.getrandbits(1))
        if update_flag:
            for g, h in user_game[u]:
                sum_for_gammaU += (
                    h
                    - alpha
                    - betaU[u]
                    - betaG[g]
                    - gammaU[u].dot(gammaG[g].transpose())[0][0]
                )
                for count_gamma in range(K):
                    diff = (
                        float(-2 * sum_for_gammaU * gammaG[g][0][count_gamma]) / total
                    )
                    gammaU[u][0][count_gamma] = gammaU[u][0][count_gamma] - rate * diff
    return gammaU


def update_gamma_g(alpha, betaU, betaG, gammaU, gammaG, rate):
    gammaG = copy.deepcopy(gammaG)
    total = len(hoursTrain)
    for g in game_user:
        sum_for_gammaG = 0
        update_flag = bool(random.getrandbits(1))
        if update_flag:
            for u, h in game_user[g]:
                sum_for_gammaG += (
                    h
                    - alpha
                    - betaU[u]
                    - betaG[g]
                    - gammaU[u].dot(gammaG[g].transpose())[0][0]
                )
                for count_gamma in range(K):
                    diff = (
                        float(-2 * sum_for_gammaG * gammaU[u][0][count_gamma]) / total
                    )
                    gammaG[g][0][count_gamma] = gammaG[g][0][count_gamma] - rate * diff
    return gammaG


def iterate_alpha_beta(alpha, betaU, betaG, gammaU, gammaG):
    pre_mse = test(hoursValid, alpha, betaU, betaG, gammaU, gammaG)
    while True:
        alpha, betaU, betaG = update_alpha_beta(5, alpha, betaU, betaG, gammaU, gammaG)
        post_mse = test(hoursValid, alpha, betaU, betaG, gammaU, gammaG)
        print(f"update alpha & beta: {post_mse}")
        if abs(post_mse - pre_mse) < 1e-5:
            return alpha, betaU, betaG
        pre_mse = post_mse


def iterate(alpha, betaU, betaG, gammaU, gammaG):
    alpha, betaU, betaG = iterate_alpha_beta(alpha, betaU, betaG, gammaU, gammaG)
    pre_mse = test(hoursValid, alpha, betaU, betaG, gammaU, gammaG)
    while True:
        gammaU = update_gamma_u(alpha, beta_u, beta_g, gammaU, gammaG, 1)
        alpha, betaU, betaG = iterate_alpha_beta(alpha, betaU, betaG, gammaU, gammaG)
        gammaG = update_gamma_g(alpha, beta_u, beta_g, gammaU, gammaG, 1)
        alpha, betaU, betaG = iterate_alpha_beta(alpha, betaU, betaG, gammaU, gammaG)
        post_mse = test(hoursValid, alpha, betaU, betaG, gammaU, gammaG)
        print(f"update gamma: {post_mse}")
        if abs(post_mse - pre_mse) < 1e-5:
            break
        pre_mse = post_mse
    return alpha, betaU, betaG, gammaU, gammaG


def test(data, alpha, betaU, betaG, gammaU, gammaG):
    preds = []
    labels = []
    for u, g, h in data:
        preds.append(
            alpha
            + betaU.get(u, numpy.average([i for i in beta_u.values()]))
            + betaG.get(g, numpy.average([i for i in beta_g.values()]))
            + gammaU[u].dot(gammaG[g].transpose())[0][0]
        )
        labels.append(h)
    return metrics.mean_squared_error(labels, preds)


alpha = numpy.mean([i[2] for i in hoursTrain])
beta_u = {}
for u in user_game:
    count_bias = 0
    count_rate = 0
    for g, h in user_game[u]:
        count_rate += 1
        count_bias += float(h - alpha)
    beta_u[u] = float(count_bias) / count_rate

beta_g = {}
for g in game_user:
    count_bias = 0
    count_rate = 0
    for u, h in game_user[g]:
        count_rate += 1
        count_bias += float(h - alpha)
    beta_g[g] = float(count_bias) / count_rate

gamma_u = {}
gamma_g = {}
K = 10
for u in allUser:
    gamma_u[u] = numpy.random.random((1, K))
for g in allGame:
    gamma_g[g] = numpy.random.random((1, K))


for u in allUser:
    for i in range(K):
        gamma_u[u][0][i] = gamma_u[u][0][i] - 0.5
        gamma_u[u][0][i] = gamma_u[u][0][i] * 0.1

for g in allGame:
    for i in range(K):
        gamma_g[g][0][i] = gamma_g[g][0][i] - 0.5
        gamma_g[g][0][i] = gamma_g[g][0][i] * 0.1

alpha, beta_u, beta_g, gamma_u, gamma_g = iterate(
    alpha, beta_u, beta_g, gamma_u, gamma_g
)

with open("predictions_Hours.csv", "w") as f:
    for l in open("pairs_Hours.csv"):
        if l.startswith("userID"):
            f.write(l)
            continue
        u, g = l.strip().split(",")

        # Logic
        pred = (
            alpha
            + beta_u.get(u, numpy.average([i for i in beta_u.values()]))
            + beta_g.get(g, numpy.average([i for i in beta_g.values()]))
            + gamma_u[u].dot(gamma_g[g].transpose())[0][0]
        )

        _ = f.write(u + "," + g + "," + str(pred) + "\n")


### Would-play
### baseline    0.6806
### now         0.696
## grade        1/2
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
for u, g, h in hoursTrain:
    user_game[u].add(g)
    game_user[g].add(u)
    user_hours[u][g] = h

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
    weight = len(user_game[u]) / averageUserVisit
    prefer_games = defaultdict(int)
    for neighbor in users_k_neighbor[u]:
        for g in user_game[neighbor]:
            prefer_games[g] += 1 / len(user_game[neighbor])
    for g in prefer_games:
        X[u][g] = prefer_games[g] / weight

sorted_X = {}
for u in allUser:
    sorted_X[u] = sorted(X[u].values(), reverse=True)

best_acc = 0
labels = [1 for _ in range(len(hoursValid))] + [0 for _ in range(len(negHoursValid))]
threshold = 560

preds = []
for u, g, _ in hoursValid + negHoursValid:
    pred = 0
    if g in X[u] and X[u][g] >= sorted_X[u][min(threshold, len(sorted_X[u]) - 1)]:
        pred = 1
    preds.append(pred)

acc = metrics.accuracy_score(labels, preds)
print(f"{acc}, {numpy.mean(preds)}")

preds = []
with open("predictions_Played.csv", "w") as f:
    for l in open("pairs_Played.csv"):
        if l.startswith("userID"):
            f.write(l)
            continue
        u, g = l.strip().split(",")
        pred = 0
        if g in X[u] and X[u][g] >= sorted_X[u][min(threshold, len(sorted_X[u]) - 1)]:
            pred = 1
        preds.append(pred)
        _ = f.write(u + "," + g + "," + str(pred) + "\n")
print(numpy.mean(preds))
