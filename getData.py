import osmnx as ox
import torch

# Lấy dữ liệu bản đồ Quận 1, TP.HCM
place_name = "Quận 1, Hồ Chí Minh, Việt Nam"
graph = ox.graph_from_place(place_name, network_type="drive")

# Chuyển đổi sang dataframe
nodes, edges = ox.graph_to_gdfs(graph)

# Trích xuất dữ liệu nút
node_data = {
    node: {
        "lat": data["y"],
        "lon": data["x"],
        "street_count": data.get("street_count", 0),
    }
    for node, data in graph.nodes(data=True)
}

# Trích xuất dữ liệu cạnh
edge_data = [
    {
        "u": u,
        "v": v,
        "length": data["length"]
    }
    for u, v, key, data in graph.edges(keys=True, data=True) if "length" in data
]

# Đóng gói và lưu
graph_data = {
    "nodes": node_data,
    "edges": edge_data
}

torch.save(graph_data, "graph_data_Q1.pth")
print("Dữ liệu đã được lưu thành công: graph_data_Q1.pth")
