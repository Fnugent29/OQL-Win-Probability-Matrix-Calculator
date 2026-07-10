import math
import numpy
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

#Team names input
#team_1_name = input("Enter the name of team 1: ")
#team_2_name = input("Enter the name of team 2: ")

team_1_name = "Demoquizzy Manifest"
team_2_name = "Legion of Zoom"

#Arrays
score_data = []
data = []
freq_data = []
name_arr = []
count_arr = []
alpha_arr = []
beta_arr = []
teams = []
removed = []
team_1_arr = []
team_2_arr = []
table_arr = []
score_cat = [0, 1, 2, 3, 4, 5, 6, 7, 8]
weeks = ["19.9", "19.8", "19.7", "19.6", "19.5", "19.4", "19.3", "19.2", "19.1",
         "18.11", "18.10", "18.9", "18.8", "18.7", "18.6", "18.5", "18.4", "18.3", "18.2", "18.1",
         "17.11", "17.10", "17.9", "17.8", "17.7", "17.6", "17.5", "17.4", "17.3", "17.2", "17.1",
         "16.11", "16.10", "16.9", "16.8", "16.7", "16.6", "16.5", "16.4", "16.3", "16.2", "16.1"]

#data formatting (Extracting number of 2's for each person)
for week in weeks:
    file = "Data" + week + ".txt"
    i = 0
    f = open(file, 'r', encoding='utf-8')
    for line in f:
        score = line.split("(")
        if len(score) == 2:
            score_clean = score[1].split(")")
            score_data.append(int(score_clean[0]))
            names = score[0].split(" ")
            names = names[0:len(names)-1]
            total_score = names[-1]
            player = ""
            for name in names:
                player = player + name + " "
            player = player[0:len(player) - 1]
            if player[-1] == "*":
                player = player[0:len(player)-1]
            j = 0
            for i in range(len(data)):
                if data[i][0] == player:
                    data[i].append(int(score_clean[0]))
                    j += 1
                    continue

            if j == 0:
                data.append([player, int(score_clean[0])])

            i += 1

#Calculating alpha and beta for the beta-binomial distributions
sample = 10
for i in range(len(data)):
    m0 = 0
    m1 = 0
    m2 = 0
    name = data[i][0]
    count = np.count_nonzero(data[i][1:])
    score_dist = []

    if count >= sample:

        for j in range(9):
            k = 0
            for score in data[i][1:sample + 1]:
                if score == j:
                    k += 1
            score_dist.append(k)

        try:
            for j in range(9):
                m1 += (score_dist[j] * score_cat[j]) / sample
                m2 += (score_dist[j] * score_cat[j] ** 2) / sample

            alpha = ((8 * m1) - m2) / (8 * ((m2 / m1) - m1 - 1) + m1)
            beta = (8 - m1) * (8 - (m2 / m1)) / (8 * ((m2 / m1) - m1 - 1) + m1)

            alpha_arr.append(alpha)
            beta_arr.append(beta)
            name_arr.append(name)
        except:
            continue

#Creating an array of team members
file = "Teams.txt"
f = open(file, 'r', encoding='utf-8')
for line in f:
    team = line.split(", ")
    teams.append(team)

for i in range(len(teams)):
    teams[i][-1] = teams[i][-1].split("\n")[0]

#Creating a list of teams
teams_dict = {}
file = "Team_names.txt"
f = open(file, "r", encoding="utf-8")
i = 0
for line in f:
    teams_dict[line] = i
    i += 1

team_1 = teams[teams_dict[team_1_name+"\n"]]
team_2 = teams[teams_dict[team_2_name+"\n"]]

dataframe = {
    "Alpha": alpha_arr,
    "Beta": beta_arr
}

#load data into a DataFrame object:
df = pd.DataFrame(dataframe, index=name_arr)

#Removing team members with an insufficient sample size
i = 0
for person in team_1:
    try:
        df.loc[person]
    except:
        removed.append(team_1[i])
    i += 1

for person in removed:
    team_1.remove(person)

removed = []
i = 0
for person in team_2:
    try:
        df.loc[person]
    except:
        removed.append(team_2[i])
    i += 1

for person in removed:
    team_2.remove(person)

#Calculating array of probabilities
pairings = numpy.zeros([len(team_1)*len(team_2), 2])
for i in range(len(team_1)):
    for j in range(len(team_2)):
        player_1 = team_1[i]
        player_2 = team_2[j]
        alpha_1 = df.loc[player_1]["Alpha"]
        beta_1 = df.loc[player_1]["Beta"]
        alpha_2 = df.loc[player_2]["Alpha"]
        beta_2 = df.loc[player_2]["Beta"]

        scale_1 = 0
        scale_2 = 0

        prob = 0
        for k in range(9):
            Q = 0
            P = abs(math.comb(8, k)*scipy.special.beta(k + alpha_2, 8 - k + beta_2)/scipy.special.beta(alpha_2, beta_2))
            for l in range(k+1, 9):
                Q += abs(math.comb(8, l)*scipy.special.beta(l + alpha_1, 8 - l + beta_1)/scipy.special.beta(alpha_1, beta_1))

            prob += P*Q

        team_1_arr.append(player_1.replace(" ", "\n"))
        team_2_arr.append(player_2.replace(" ", "\n"))
        table_arr.append(np.round(100*prob, 2))

#Assembling array of probabilities into a dataframe
dataframe_2 = {
    "Team 1": team_1_arr,
    "Team 2": team_2_arr,
    "Prob": table_arr
}

df_2 = pd.DataFrame(dataframe_2).pivot(index="Team 1", columns="Team 2", values="Prob")

#Creating probability grid
sns.heatmap(df_2, annot=True, cmap="RdYlGn", vmin=0, vmax=100, fmt=".1f", linewidth=.5)
plt.title("Probability (%) of each player on " + team_1_name + " getting more \n 2's than each player on " + team_2_name)
plt.xticks(rotation=0)
plt.xlabel(team_2_name)
plt.ylabel(team_1_name)
plt.show()

