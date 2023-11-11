import gzip
from collections import defaultdict


def readJSON(path):
    for l in gzip.open(path, "rt"):
        d = eval(l)
        u = d["userID"]
        try:
            g = d["gameID"]
        except Exception as e:
            g = None
        yield u, g, d

allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)

hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

user_game = defaultdict(set)
game_user = defaultdict(set)
for i in hoursTrain:
    user_game[i[0]].add(i[1])
    game_user[i[1]].add(i[0])



### Time-played 
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
    alpha = iterate(5, alpha, betaU, betaI)

predictions = open("predictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    pred = alpha + betaU[u] + betaI[g]

    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()

### Would-play
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


gameCount = defaultdict(int)
allGame = set()
for i in allHours:
    gameCount[i[1]] += 1
    allGame.add(i[1])
totalPlayed = len(allHours)
mostPopular = sorted([(gameCount[x], x) for x in gameCount], reverse=True)
predictions = open("predictions_Played.csv", "w")
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u, g = l.strip().split(",")

    count = 0
    return1 = set()
    limit = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        limit = ic
        if count > totalPlayed * 0.7:
            break
    feat = jaccard_feat(u, g)
    if feat > 0.06:
        pred = 1
    elif g in return1 and feat > 0.015:
        pred = 1

    _ = predictions.write(u + "," + g + "," + str(pred) + "\n")

predictions.close()
