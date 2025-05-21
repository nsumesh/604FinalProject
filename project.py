import pandas as pd
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt

def solve(t):
    players = pd.read_csv("/Users/nsumesh/Desktop/UMD/Spring 2025/MSML 604/Sorted_price.csv")
    players.columns = ["Player", "Score", "Price", "Role", "Foreigner"]

    scores = players["Score"].values
    prices = players["Price"].values
    roles = players["Role"].values
    foreigners = players["Foreigner"].values
    n = len(players)

    P = np.load("/Users/nsumesh/Desktop/UMD/Spring 2025/MSML 604/P_v2.npy")

    model = ConcreteModel()
    model.players = RangeSet(0, n - 1)
    model.x = Var(model.players, within=Binary)

    model.obj = Objective(
        expr= - (sum(P[i][j] * model.x[i] * model.x[j] for i in model.players for j in model.players) +
                 t * sum(scores[i] * model.x[i] for i in model.players)),
        sense=minimize
    )

    model.total_players = Constraint(expr=sum(model.x[i] for i in model.players) == 11)
    model.budget = Constraint(expr=sum(prices[i] * model.x[i] for i in model.players) <= 100)

    model.min_bats = Constraint(expr=sum(model.x[i] for i in model.players if roles[i] == 'Batsman') >= 3)
    model.max_bats = Constraint(expr=sum(model.x[i] for i in model.players if roles[i] == 'Batsman') <= 6)

    model.min_bowl = Constraint(expr=sum(model.x[i] for i in model.players if roles[i] == 'Bowler') >= 3)
    model.max_bowl = Constraint(expr=sum(model.x[i] for i in model.players if roles[i] == 'Bowler') <= 6)

    model.max_foreign = Constraint(expr=sum(model.x[i] for i in model.players if foreigners[i] == 1) <= 4)

    solver = SolverFactory("gurobi")
    results = solver.solve(model, tee=True)

    selected_players = [i for i in model.players if model.x[i].value == 1]
    selected_players_csv = players.iloc[selected_players]
    cost = selected_players_csv["Price"].sum()
    score = selected_players_csv["Score"].sum()
    selected_players_sorted = selected_players_csv.sort_values(
        by="Role", ascending=False,
        key=lambda col: col.map({'Batsman': 1, 'Bowler': 0})
    )

    return cost, score, selected_players_sorted

t_s = []
costs = []
scores = []

for t in np.linspace(0, 0.01, 21):
    cost, score, selected_players_sorted = solve(t)
    t_s.append(t)
    costs.append(cost)
    scores.append(score)

plt.plot(t_s, costs, marker='o', label='Cost')
plt.xlabel('t')
plt.ylabel('Total Cost')
plt.title('Cost vs t')
plt.grid()
plt.legend()
plt.show()

plt.plot(t_s, scores, marker='o', label='Score', color='green')
plt.xlabel('t')
plt.ylabel('Total Score')
plt.title('Score vs t')
plt.grid()
plt.legend()
plt.show()

print("\nSelected Fantasy Team :")
print(selected_players_sorted.to_string(index=False))

print("\nCosts:", costs)
print("Scores:", scores)
