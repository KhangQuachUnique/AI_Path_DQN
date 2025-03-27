import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# ---------------------------------------------
# 1. Lấy dữ liệu bản đồ từ OpenStreetMap
# ---------------------------------------------
print("Lấy dữ liệu OSM...")
# Ví dụ: lấy dữ liệu của Quận 1, TP.HCM
place_name = "Quận 1, Hồ Chí Minh, Việt Nam"
graph = ox.graph_from_place(place_name, network_type="drive")
# Lấy danh sách node (node ID thực tế từ OSM)
node_ids = list(graph.nodes)
# Tạo dictionary chuyển đổi: node_id -> index (sử dụng trong vector one-hot)
node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
num_nodes = len(node_ids)
print(f"Số node: {num_nodes}")

# Xây dựng ma trận kề (chú ý: dùng numpy array, có giá trị 0/1)
adj_matrix = nx.to_numpy_array(graph, nodelist=node_ids)
# Với môi trường định tuyến, ta cần lấy thông tin các neighbor của mỗi node
neighbors = {node: list(graph.neighbors(node)) for node in node_ids}

# ---------------------------------------------
# 2. Xây dựng môi trường định tuyến đơn giản
# ---------------------------------------------
# Môi trường này sẽ:
# - Trạng thái: one-hot vector kích thước num_nodes (chỉ số hiện tại = 1)
# - Hành động: chọn 1 trong số các node kề của node hiện tại
# - Reward: +100 nếu đến đích, -1 cho mỗi bước di chuyển
# - Episode kết thúc nếu đạt đích hoặc bước di chuyển vượt quá giới hạn

class RouteEnv:
    def __init__(self, graph, node_ids, neighbors):
        self.graph = graph
        self.node_ids = node_ids
        self.neighbors = neighbors
        self.num_nodes = len(node_ids)
        self.current_node = None
        self.goal = None
        self.max_steps = 100

    def reset(self, start_node, goal_node):
        """Khởi tạo lại môi trường với start và goal (node ID)"""
        self.current_node = start_node
        self.goal = goal_node
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        # Sử dụng one-hot vector cho trạng thái
        state = np.zeros(self.num_nodes, dtype=np.float32)
        index = node_id_to_index[self.current_node]
        state[index] = 1.0
        return state

    def step(self, action_node):
        """Nhận hành động là node ID mà agent chọn di chuyển tới.
           Nếu action_node không nằm trong neighbor của current_node, ta xem như bước đi không hợp lệ và giữ nguyên vị trí.
        """
        self.steps += 1
        if action_node in self.neighbors[self.current_node]:
            self.current_node = action_node
        else:
            # Nếu hành động không hợp lệ, ta có thể phạt thêm
            pass

        done = False
        reward = -1.0
        if self.current_node == self.goal:
            reward = 100.0
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done

# ---------------------------------------------
# 3. Xây dựng DQN với PyTorch
# ---------------------------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

# Ở đây, state_dim = num_nodes.
# Vì môi trường cho phép hành động là di chuyển đến các node kề, action_dim = num_nodes.
state_dim = num_nodes
action_dim = num_nodes

# Khởi tạo mô hình và target network
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# ---------------------------------------------
# 4. Replay Buffer
# ---------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

buffer = ReplayBuffer()

# ---------------------------------------------
# 5. Hàm chọn hành động theo ε-greedy
# ---------------------------------------------
def select_action(state, current_node, env, epsilon):
    # Lấy danh sách hành động hợp lệ từ nút hiện tại
    valid_actions = env.neighbors.get(current_node, [])
    if len(valid_actions) == 0:
        # Nếu không có hành động hợp lệ, giữ nguyên vị trí hiện tại
        return current_node
    
    if random.random() < epsilon:
        return random.choice(valid_actions)
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape: [1, state_dim]
        with torch.no_grad():
            q_values = policy_net(state_tensor).squeeze(0).numpy()
        # Lấy Q-values của các hành động hợp lệ
        valid_q = {node: q_values[node_id_to_index[node]] for node in valid_actions if node in node_id_to_index}
        if len(valid_q) == 0:
            return random.choice(valid_actions)
        best_action = max(valid_q, key=valid_q.get)
        return best_action

# ---------------------------------------------
# 6. Hàm huấn luyện DQN
# ---------------------------------------------
def optimize_model(batch_size, gamma):
    if len(buffer) < batch_size:
        return
    batch = buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.tensor(np.array(states), dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    
    # Tính Q(s,a) từ policy network
    state_action_values = policy_net(states)
    
    # Ta chỉ cần lấy Q-value của hành động mà agent đã thực hiện:
    # Vì hành động là node ID, ta chuyển sang index
    action_indices = torch.tensor([node_id_to_index[a] for a in actions], dtype=torch.long)
    state_action_values = state_action_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
    
    # Tính giá trị Q kỳ vọng từ target network:
    with torch.no_grad():
        next_state_values = target_net(next_states).max(1)[0]
    expected_state_action_values = rewards + gamma * next_state_values * (1 - dones)
    
    loss = loss_fn(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ---------------------------------------------
# 7. Huấn luyện qua nhiều episode
# ---------------------------------------------
env = RouteEnv(graph, node_ids, neighbors)

num_episodes = 1000  # Số episode huấn luyện
batch_size = 64
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995  # giảm epsilon mỗi episode
update_target_every = 10  # cập nhật target network mỗi 10 episode

# Chọn ngẫu nhiên start và goal từ danh sách node (đảm bảo khác nhau)
def sample_start_goal():
    start = random.choice(node_ids)
    goal = random.choice(node_ids)
    while goal == start:
        goal = random.choice(node_ids)
    return start, goal

print("Bắt đầu huấn luyện...")
epsilon = epsilon_start
for episode in range(1, num_episodes+1):
    start_node, goal_node = sample_start_goal()
    state = env.reset(start_node, goal_node)
    total_reward = 0.0

    for t in range(env.max_steps):
        action = select_action(state, env.current_node, env, epsilon)
        next_state, reward, done = env.step(action)
        total_reward += reward
        buffer.push(state, action, reward, next_state, done)
        state = next_state

        optimize_model(batch_size, gamma)
        if done:
            break
    
    # Giảm epsilon dần
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # Cập nhật target network định kỳ
    if episode % update_target_every == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode} - Tổng reward: {total_reward} - Số bước: {t+1}")

print("Huấn luyện hoàn tất!")

# ---------------------------------------------
# 8. Thử nghiệm: Tìm đường từ một điểm cố định
# ---------------------------------------------
# Chọn start và goal cố định (lấy 2 node khác nhau)
start_node = node_ids[0]
goal_node = node_ids[10] if node_ids[10] != start_node else node_ids[11]
print(f"Thử nghiệm: start = {start_node}, goal = {goal_node}")

state = env.reset(start_node, goal_node)
route = [env.current_node]

# Giới hạn số bước tìm đường
for _ in range(100):
    action = select_action(state, env.current_node, env, epsilon=0.0)  # Không random
    state, reward, done = env.step(action)
    route.append(env.current_node)
    if done:
        break

# Kiểm tra xem route có hợp lệ không
missing_nodes = [n for n in route if n not in graph.nodes]
if missing_nodes:
    raise ValueError(f"Các node sau không tồn tại trong graph: {missing_nodes}")
else:
    print("Route tìm được:", route)

# ---------------------------------------------
# 9. Vẽ đường đi trên bản đồ
# ---------------------------------------------
fig, ax = ox.plot_graph_route(graph, route, route_linewidth=4, node_size=50, bgcolor="w")
plt.show()
