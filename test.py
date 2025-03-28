import osmnx as ox
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import folium
import math
import pickle

# Đặt seed để tái lập kết quả (nếu cần)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = "cpu"

# ---------------------------------------------
# 1. Lấy dữ liệu bản đồ từ OpenStreetMap
# ---------------------------------------------
print("Lấy dữ liệu OSM...")
# Lấy dữ liệu bản đồ quanh một điểm (ví dụ: quanh tọa độ đã cho)
latitude, longitude = 10.782222, 106.695833
graph = ox.graph_from_point((latitude, longitude), dist=200, network_type="walk")

node_ids = list(graph.nodes)
node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
num_nodes = len(node_ids)
print(f"Số node: {num_nodes}")

# Ma trận kề (không dùng trực tiếp nhưng có thể hữu ích)
adj_matrix = nx.to_numpy_array(graph, nodelist=node_ids)
# Lấy thông tin các node kề (neighbors) từ graph
neighbors = {node: list(graph.neighbors(node)) for node in node_ids}
print(neighbors[411925946])

# ---------------------------------------------
# 2. Xây dựng môi trường định tuyến với state mở rộng:
#    - one-hot vector kích thước num_nodes
#    - Thêm thông tin góc và khoảng cách đến đích
# ---------------------------------------------
class RouteEnv:
    def __init__(self, graph, node_ids, neighbors):
        self.graph = graph
        self.node_ids = node_ids
        self.neighbors = neighbors
        self.num_nodes = len(node_ids)
        self.current_node = None
        self.goal = None
        self.max_steps = 60
        self.visited = set()

    def reset(self, start_node, goal_node):
        """Khởi tạo lại môi trường với start và goal (node ID)"""
        self.current_node = start_node
        self.goal = goal_node
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        # Tạo one-hot vector cho state: kích thước num_nodes
        state = np.zeros(self.num_nodes, dtype=np.float32)
        start_index = node_id_to_index[self.current_node]
        goal_index = node_id_to_index[self.goal]
        state[start_index] = 1.0
        state[goal_index] = 2.0

        # Lấy tọa độ của current_node và goal
        curr_coord = np.array([self.graph.nodes[self.current_node]['x'], self.graph.nodes[self.current_node]['y']])
        goal_coord = np.array([self.graph.nodes[self.goal]['x'], self.graph.nodes[self.goal]['y']])
        
        # Tính khoảng cách Euclid và góc (radian) từ current_node đến goal
        distance = np.linalg.norm(goal_coord - curr_coord)
        dx = goal_coord[0] - curr_coord[0]
        dy = goal_coord[1] - curr_coord[1]
        angle = math.atan2(dy, dx)
        
        # Nối thêm 2 đặc trưng (angle, distance) vào state
        state = np.concatenate([state, np.array([angle, distance], dtype=np.float32)])
        return state

    def step(self, action_node):
        """
        Nhận hành động là node ID mà agent chọn di chuyển tới.
        Nếu action_node không nằm trong neighbor của current_node, giữ nguyên vị trí và phạt thêm.
        Reward được tính dựa trên cải thiện khoảng cách đến đích, trừ phạt mỗi bước và phạt hành động không hợp lệ.
        """
        self.steps += 1

        # Tính khoảng cách trước khi di chuyển
        curr_coord = np.array([self.graph.nodes[self.current_node]['x'], self.graph.nodes[self.current_node]['y']])
        goal_coord = np.array([self.graph.nodes[self.goal]['x'], self.graph.nodes[self.goal]['y']])
        prev_distance = np.linalg.norm(goal_coord - curr_coord)

        # Kiểm tra hành động có hợp lệ hay không
        valid = action_node in self.neighbors[self.current_node]
        if valid:
            self.current_node = action_node
            self.visited.add(action_node)
        # Nếu không hợp lệ thì giữ nguyên vị trí

        # Tính khoảng cách sau khi di chuyển
        new_coord = np.array([self.graph.nodes[self.current_node]['x'], self.graph.nodes[self.current_node]['y']])
        new_distance = np.linalg.norm(goal_coord - new_coord)

        # Nếu đến đích
        if self.current_node == self.goal:
            reward = 250.0
            done = True
            self.visited = set()
        else:
            # Reward dựa trên sự cải thiện khoảng cách (prev_distance - new_distance)
            improvement = prev_distance - new_distance
            reward = improvement * 10.0  # hệ số khuếch đại cải thiện
            reward -= 1.0                # phạt mỗi bước đi
            if not valid:
                reward -= 5.0            # phạt nếu hành động không hợp lệ
            if self.current_node in self.visited:
                reward -= 5.0            # phạt nếu quay lại node đã đi qua4

            done = False
            if self.steps >= self.max_steps:
                done = True
                self.visited = set()

        return self._get_state(), reward, done

# ---------------------------------------------
# 3. Xây dựng mô hình DQN với PyTorch
# ---------------------------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),  # Tăng số neuron để học được biểu diễn tốt hơn
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

# ---------------------------------------------
# 4. Replay Buffer
# ---------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------
# 5. Hàm chọn hành động theo ε-greedy
# ---------------------------------------------
def select_action(state, current_node, env, epsilon):
    # Lấy danh sách hành động hợp lệ từ nút hiện tại
    valid_actions = env.neighbors.get(current_node, [])
    if len(valid_actions) == 0:
        return current_node
    
    if random.random() < epsilon:
        return random.choice(valid_actions)
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor).squeeze(0).cpu().numpy()
        # Lấy Q-value cho các hành động hợp lệ
        valid_q = {node: q_values[node_id_to_index[node]] for node in valid_actions if node in node_id_to_index}
        if len(valid_q) == 0:
            return random.choice(valid_actions)
        best_action = max(valid_q, key=valid_q.get)
        return best_action

# ---------------------------------------------
# 6. Hàm huấn luyện DQN (optimize model)
# ---------------------------------------------
def optimize_model(batch_size, gamma):
    if len(buffer) < batch_size:
        return
    batch = buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    action_indices = torch.tensor([node_id_to_index[a] for a in actions], dtype=torch.long, device=device)
    state_action_values = policy_net(states).gather(1, action_indices.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        next_state_values = target_net(next_states).max(1)[0]
    expected_state_action_values = rewards + gamma * next_state_values * (1 - dones)
    
    loss = loss_fn(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ---------------------------------------------
# 7. Khởi tạo các biến và model
# ---------------------------------------------
# Vì state có thêm 2 đặc trưng (angle và distance) nên state_dim = num_nodes + 2
state_dim = num_nodes + 2
action_dim = num_nodes  # Hành động: chọn chuyển đến 1 trong các node

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

buffer = ReplayBuffer()
env = RouteEnv(graph, node_ids, neighbors)

# ---------------------------------------------
# 8. Các hàm lưu / load model và replay buffer
# ---------------------------------------------
def save_model(path="dqn_model.pth"):
    torch.save(policy_net.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path="dqn_model.pth"):
    if os.path.exists(path):
        policy_net.load_state_dict(torch.load(path, map_location=device))
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Model loaded from {path}")
    else:
        print("No model found, training from scratch.")

def save_buffer(path="replay_buffer.pkl"):
    with open(path, "wb") as f:
        pickle.dump(buffer, f)
    print(f"Replay buffer saved to {path}")

def load_buffer(path="replay_buffer.pkl"):
    global buffer
    if os.path.exists(path):
        with open(path, "rb") as f:
            buffer = pickle.load(f)
        print(f"Replay buffer loaded from {path}")
    else:
        print("No replay buffer found, starting fresh.")

# ---------------------------------------------
# 9. Menu chính với các tùy chọn Train, Test, Lưu, Load...
# ---------------------------------------------
while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    print("1. Load model")
    print("2. Load buffer")
    print("3. Train model")
    print("4. Test model")
    print("5. Save model")
    print("6. Save buffer")
    print("7. Exit")
    choice = input("Chọn chức năng: ")
    if choice == "7":
        break
    elif choice == "1":
        load_model()
    elif choice == "3":
        num_episodes = int(input("Nhập số episode huấn luyện: "))
        batch_size = 128
        gamma = 0.99
        epsilon_start = 1.0
        epsilon_end = 0.2
        epsilon_decay = (epsilon_start - epsilon_end) / (num_episodes)  # Giảm epsilon đều theo số episode
        update_target_every = 10  # Cập nhật target network mỗi 10 episode

        print("Bắt đầu huấn luyện...")
        epsilon = epsilon_start
        reached_goal_rate = 0.0
        for episode in range(1, num_episodes + 1):
            # Chọn start và goal ngẫu nhiên, đảm bảo khác nhau
            start_node = random.choice(node_ids)
            goal_node = random.choice(node_ids)
            while goal_node == start_node:
                goal_node = random.choice(node_ids)
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
            
            # Giảm epsilon sau mỗi episode
            epsilon = max(epsilon_end, epsilon - epsilon_decay)

            # Cập nhật target network định kỳ
            if episode % update_target_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Ghi nhận số lần đạt đích (dựa trên tổng reward dương)
            if total_reward > 0:
                reached_goal_rate += 1.0

            print(f"Episode {episode} - Tổng reward: {total_reward:.2f} - Số bước: {t+1} - Epsilon: {epsilon:.3f}")

            if episode % 100 == 0:
                print(f"Goal rate: {reached_goal_rate/100*100}%")
                reached_goal_rate = 0.0
        print("Huấn luyện hoàn tất!")
    elif choice == "2":
        load_buffer()
    elif choice == "4":
            try:
                # Cho người dùng nhập chỉ số node (index trong node_ids)
                start = int(input("Nhập node xuất phát (index): "))
                goal = int(input("Nhập node đích (index): "))
                start_node = node_ids[start]
                goal_node = node_ids[goal]
                if start_node == goal_node:
                    print("Start và goal không được trùng nhau!")
                    continue
                print(f"Thử nghiệm: start = {start_node}, goal = {goal_node}")

                state = env.reset(start_node, goal_node)
                route = [env.current_node]

                for _ in range(100):
                    action = select_action(state, env.current_node, env, epsilon=0.0)  # Không random khi test
                    state, reward, done = env.step(action)
                    route.append(env.current_node)
                    if done:
                        print("Đã đến đích!")
                        break

                missing_nodes = [n for n in route if n not in graph.nodes]
                if missing_nodes:
                    raise ValueError(f"Các node sau không tồn tại trong graph: {missing_nodes}")
                else:
                    print("Route tìm được:", route)

                route_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]
                m = folium.Map(location=route_coords[0], zoom_start=16)
                folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.7).add_to(m)
                folium.Marker(route_coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
                goal_coords = (graph.nodes[goal_node]['y'], graph.nodes[goal_node]['x'])
                folium.Marker(goal_coords, popup="Goal", icon=folium.Icon(color="red")).add_to(m)
                m.save("route_map.html")
                input("Nhấn Enter để tiếp tục...")
            except Exception as e:
                print(f"Lỗi: {e}")
    elif choice == "5":
        save_model()
    elif choice == "6":
        save_buffer()
