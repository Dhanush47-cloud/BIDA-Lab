import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("data_exp9.txt", sep=r"\s+", header=None,
                 names=["user_1", "user_2"])

# 2. Create social network graph
G = nx.from_pandas_edgelist(df, "user_1", "user_2")

print("Number of Users:", G.number_of_nodes())
print("Number of Connections:", G.number_of_edges())

# 3. Calculate centrality measures
degree = nx.degree_centrality(G)

# Approximate betweenness centrality for faster execution
betweenness = nx.betweenness_centrality(G, k=200, seed=42)

closeness = nx.closeness_centrality(G)

# 4. Display top 5 influential users
print("\nDegree Centrality:")
print(sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5])

print("\nBetweenness Centrality:")
print(sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5])

print("\nCloseness Centrality:")
print(sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5])

# 5. Visualize a smaller network (200 users)
nodes = list(G.nodes())[:200]
H = G.subgraph(nodes)

pos = nx.spring_layout(H, seed=42, iterations=10)

nx.draw(H, pos, with_labels=True, node_size=100,
        node_color="skyblue", font_size=6, edge_color="gray")

plt.title("Social Network Analysis")
plt.show()
