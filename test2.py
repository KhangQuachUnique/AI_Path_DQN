import pandas as pd 
import torch 
from torch_geometric.data import Data
import osmnx as ox
import time
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# G = ox.load_graphml('data/HCMC_Quan1.graphml')

# # Trích xuất danh sách cạnh với thông tin u, v, length
# edges_data = [
#     (u, v, data.get("length", 0))  # Nếu không có "length", mặc định là 0
#     for u, v, data in G.edges(data=True)
# ]

# nodes_data = [
#     (node, data.get("x"), data.get("y"), data.get("street_count"))
#     for node, data in G.nodes(data=True)
# ]

# Chuyển dữ liệu thành DataFrame
# edges = pd.DataFrame(edges_data, columns=["u", "v", "length"])
# nodes = pd.DataFrame(nodes_data, columns=["node_id", "x", "y", "street_count"])
# # Lưu vào file CSV

# edges.rename(columns={"u": "head", "v": "tail"}, inplace=True)
# edges.to_csv('edges_Q1.csv', index=False)
# nodes.to_csv('nodes_Q1.csv', index=True)


nodes = pd.read_csv('nodes_Q1.csv')
edges = pd.read_csv('edges_Q1.csv')


# node_ids = nodes["node_id"].unique()
# id_to_index = {id: i for i, id in enumerate(node_ids)}

# # Chuyển node_id thành index liên tục
# nodes["index"] = nodes["node_id"].map(id_to_index)
# edges["head"] = edges["head"].map(id_to_index)
# edges["tail"] = edges["tail"].map(id_to_index)

# nodes.to_csv('nodes_Q1.csv', index=False)
# edges.to_csv('edges_Q1.csv', index=False)
# print(nodes["node_id"])
# print(edges[["head","tail"]])

# nodes = nodes.drop(columns=["highway"])
# nodes.to_csv('nodes.csv', index= False)

# edges = edges.drop(columns=['oneway','highway'])
# edges["geometry"] = edges["geometry"].apply(ast.literal_eval)
# edges.to_csv('edges.csv',index=False)

# print(type(edges["geometry"].values))

# print(nodes[["x","y","street_count"]].values)










x = torch.tensor(nodes[["x","y","street_count"]].values, dtype=torch.float32)
# print(node_features)

edges_index = torch.tensor(edges[["head","tail"]].values.T, dtype=torch.long)

# edges["geometry"] = edges["geometry"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
# print(type(edges["geometry"].values[0][0][0]))

edges_attr = torch.tensor(edges[["length"]].values, dtype=torch.float32)

graph = Data(x=x, edge_index=edges_index, edge_attr=edges_attr)

torch.save(graph, "graph_data_Q1.pth")
print("Save successed")
print(graph.x)
print(graph.edge_index)
print(graph.edge_attr)
num_nodes = x.shape[0]
max_index = edges_index.max().item()
min_index = edges_index.min().item()

print(f"Number of nodes: {num_nodes}")
print(f"Max index in edge_index: {max_index}")
print(f"Min index in edge_index: {min_index}")

class GNN(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return(x)
    
print(graph)
print(graph.x.shape)  # Số lượng node và số feature trên mỗi node
print(graph.edge_index.shape)  # Kích thước của danh sách cạnh
print(graph.edge_attr.shape)

device = "cuda"

model = GNN(in_features=3, hidden_dim=16, out_features=3).to(device)

graph = graph.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def compute_loss(model, graph):
    out = model(graph.x, graph.edge_index)

    anchor = out[graph.edge_index[0]]
    positive = out[graph.edge_index[1]]

    negative_indices = torch.randint(0, graph.x.shape[0], (positive.shape[0],), device=device)  
    negative = out[negative_indices]

    triplet_loss = F.triplet_margin_loss(anchor, positive, negative, margin=1.0)

    cosine_loss = F.cosine_embedding_loss(out, graph.x, torch.ones(graph.x.shape[0], device=device))

    total_loss = triplet_loss + 0.5 * cosine_loss
    return total_loss

start_time = time.time()

model.train()
for epoch in range(10000):
    optimizer.zero_grad()
    loss = compute_loss(model, graph)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

end_time = time.time()
print(f"time_train: {end_time - start_time}")
torch.save(model.state_dict(), "gnn_model_Q1.pth")
model.eval()
with torch.no_grad():
    out = model(graph.x, graph.edge_index)
print(out[:30].cpu())

export GNN