import random
import sys
import matplotlib.pyplot as plt
import math
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation, FeatureAgglomeration, SpectralClustering, \
    MiniBatchKMeans, Birch, DBSCAN, OPTICS, AgglomerativeClustering
from torch_geometric.nn import GCNConv, SAGEConv, GAE, GINConv, GATConv
from torch_geometric.utils import train_test_split_edges, to_networkx, from_networkx, to_dense_adj
from torch_geometric.transforms import NormalizeFeatures, ToDevice, RandomLinkSplit, RemoveDuplicatedEdges
import os
import torch
from torch_geometric.data import Data
import os.path as osp
from torch.utils.data import Dataset
import pandas as pd


#####################################################################################################################
os.environ['TCL_LIBRARY'] = r'C:\Users\zyfel\AppData\Local\Programs\Python\Python312\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\zyfel\AppData\Local\Programs\Python\Python312\tcl\tk8.6'

current_directory = os.getcwd()

print("Current working directory:", current_directory)

# Set the random seed for reproducibility
seed = 42
run_index = sys.argv[1]

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#####################################################################################################################


# Define a function to parse each line of data
def parse_data(line):
    parts = line.split()
    return {
        'RSU': int(parts[1]),
        'CAR': int(parts[3]),
        'X': float(parts[5]),
        'Y': float(parts[7]),
        'dBm': float(parts[9]),
        'ReceiveTime': float(parts[11].split(':')[1]) if ':' in parts[11] else float(parts[11]),
        'SentTime': float(parts[13])
    }


# Read data from a text file
file_path = './data/data.txt'  # Replace with your file path
with open(file_path, 'r') as file:
    lines = file.readlines()

# Parse the data
parsed_data = []
for line in lines:
    try:
        parsed_data.append(parse_data(line.strip()))
    except Exception as e:
        print(f"Error parsing line: {line}. Error: {e}")

# Create a DataFrame
df = pd.DataFrame(parsed_data)

# Remove redundant data (duplicate rows based on RSU, CAR, X, Y)
df_unique = df.drop_duplicates(subset=['RSU', 'CAR', 'X', 'Y'])
print(df_unique.columns)
# Sort the DataFrame by the 'ReceiveTime' column in increasing order
df_sorted = df_unique.sort_values(by='ReceiveTime')

# Optionally, save the sorted DataFrame to a CSV file
df_sorted.to_csv(f'./data/one-hot/gnn_kmeans_results/sorted_data/sorted_data_time{run_index}.csv', index=False)

with open(file_path, 'w') as file:
    pass

#####################################################################################################################


# Load data
data = pd.read_csv('./data/node_data.csv')
df = data.reset_index(drop=True)

# Initialize columns for one-hot encoding (r1 to r30)
for i in range(1, 31):
    df[f'r{i}'] = 0.0

# Calculate distances and update the nearest RSU
for i in range(len(df)):
    cur_x = df.loc[i, 'x']
    cur_y = df.loc[i, 'y']
    index = 0
    dis = float('inf')
    for j in range(30):  # Assuming there are 30 RSUs
        rsu_x = df.loc[j, 'x']
        rsu_y = df.loc[j, 'y']
        dis_temp = math.sqrt((cur_x - rsu_x) ** 2 + (cur_y - rsu_y) ** 2)
        if dis_temp < dis:
            index = j
            dis = dis_temp
    df.loc[i, f'r{index+1}'] += 4000  # Update the nearest RSU's one-hot column
    print(f"Processed node {i+1}/{len(df)}")

# Save the DataFrame with one-hot features
df.to_csv('./data/node_data+rsu.csv', index=False)


#####################################################################################################################


data = pd.read_csv(f'./data/one-hot/gnn_kmeans_results/sorted_data/sorted_data_time{run_index}.csv')
data = data.reset_index(drop=True)

df = pd.read_csv('./data/node_data+rsu.csv')
df = df.reset_index(drop=True)
df = df.assign(car=0.0)

data = data.drop_duplicates(subset=['RSU', 'CAR'])
data.reset_index(drop=True, inplace=True)
print(data.index)

for i in range(len(data)):
    car_pos = (data.loc[i, 'X'],  data.loc[i, 'Y'])
    index = 0
    dis = 9999999999
    for j in range(len(df)):
        node_pos = (df.loc[j, 'x'],  df.loc[j, 'y'])
        temp = math.dist(car_pos, node_pos)
        if temp < dis:

            dis = temp
            index = j
    df.loc[index, 'car'] += 100
    print(i)

df.to_csv('./data/node_data+rsu+car.csv', index=False)

#####################################################################################################################

# Define the root directory where the dataset will be stored
root = './'
version = 'v1'
run_id = 'test'

# Define the Number of Clusters
num_clusters = 30
n = 10  # epoch and n_init refers to the number of times the clustering algorithm will run different initializations
clusters = []

# Transform Parameters
transform_set = True
value_num = 0.1
test_num = 0.2

# Optimizer Parameters (learning rate)
learn_rate = 0.01

# Epochs or the number of generation/iterations of the training dataset
epochs = 500

# Setting up Colours for the test visualizations
cmap = plt.get_cmap('tab20')
colors_full = [cmap(i) for i in np.linspace(0, 1, 30)]


#####################################################################################################################


df = pd.read_csv('./data/node_data+rsu+car.csv')
df = df.reset_index(drop=True)

eg = pd.read_csv('./data/edge_data.csv')
eg = eg.reset_index(drop=True)


#####################################################################################################################


class MyDataset(Dataset):
    def __init__(self, root, file_name, transform=None, pre_transform=None):
        self.root = root
        self.filename = file_name
        self.transform = transform
        self.pre_transform = pre_transform
        self.raw_dir = osp.join(root, 'raw')

        # Ensure raw data is available
        self._download()
        # Load data paths
        self.node_data_path = osp.join(self.raw_dir, self.filename[0])
        self.edge_data_path = osp.join(self.raw_dir, self.filename[1])
        # Load dataframes
        self.node_data = pd.read_csv(self.node_data_path)
        self.edge_data = pd.read_csv(self.edge_data_path)
        self.edge_data.reset_index(drop=True, inplace=True)

    def _download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        from_path0 = os.path.join(self.root, str(self.filename[0]))
        to_path0 = os.path.join(self.raw_dir, str(self.filename[0]))
        if not osp.exists(to_path0):
            df = pd.read_csv(from_path0)
            df.to_csv(to_path0, index=False)

        from_path1 = os.path.join(self.root, str(self.filename[1]))
        to_path1 = os.path.join(self.raw_dir, str(self.filename[1]))
        if not osp.exists(to_path1):
            eg = pd.read_csv(from_path1)
            eg.reset_index(drop=True, inplace=True)
            eg.to_csv(to_path1, index=False)

    def __len__(self):
        return len(self.node_data)

    def __getitem__(self, idx):
        node_features = self.node_data.drop(columns=['rid']).values.astype(float)
        x = torch.tensor(node_features, dtype=torch.float)  # Keep all features

        edge_source = self.edge_data['node1'].values.astype(int)
        edge_target = self.edge_data['node2'].values.astype(int)

        edge_index = np.array([edge_source, edge_target])
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index
        )

        if self.transform:
            data = self.transform(data)

        return data


#####################################################################################################################


# Define the root directory where the dataset will be stored
root = './data'
file_name = ['node_data+rsu+car.csv', 'edge_data.csv']

dataset_ = MyDataset(root, file_name)
dataset = MyDataset(root, file_name, transform=NormalizeFeatures())
print(dataset[0])
print('Done')

data_ = dataset_[0]
print(data_)

G = to_networkx(data_)
G = G.to_undirected()

X = data_.x[:, [0, 1]].cpu().detach().numpy()
pos = dict(zip(range(X[:, 0].size), X))

fig, ax = plt.subplots(figsize=(10, 10))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = RandomLinkSplit(
    num_val=0.1, num_test=0.3,
    is_undirected=True,
    split_labels=True
)

data = dataset[0]
train_data, val_data, test_data = transform(data)


#####################################################################################################################


# Create adjacency matrix
A = {}
edge_index = train_data.edge_index
for i in range(edge_index.shape[1]):
    src = edge_index[0, i].item()
    tgt = edge_index[1, i].item()
    if src not in A:
        A[src] = []
    A[src].append(tgt)

# print('------------')
train_pos_edge_index = train_data.pos_edge_label_index
# print('training positive edges:', train_pos_edge_index)
# print('------------')
# print(f'Number of graphs: {len(dataset)}')
# print(f'Number of features: {dataset.num_features}')
# print('------------')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#####################################################################################################################


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        #Defines the GNN Layers
        print('in channels', in_channels)
        print('out channels', out_channels)
        self.conv1 = SAGEConv(in_channels, out_channels)
        self.conv2 = SAGEConv(out_channels, in_channels)

    def forward(self, x, edge_index):
        x = x.float()
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


"""define autoencoder"""

# Channel Parameters & GAE MODEL
in_channels = data.num_features
out_channels = 64
print('features', data.num_features, 'edges', out_channels)

# Initialize the model
model = GAE(GCNEncoder(in_channels, out_channels))

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

model = model.to(device).float()
data = dataset[0].to(device)
x = train_data.x.to(device).float()
# train_pos_edge_index = train_data.pos_edge_label_index.to(device)
train_pos_edge_index = train_pos_edge_index.to(device)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss), z


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        x = data.x.to(device)
        pos_edge_index = pos_edge_index.to(device)
        z = model.encode(x, pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


# print(model)


#####################################################################################################################



auc_values = []
ap_values = []

best_auc = 0.0  # Track the best AUC value
consecutive_epochs = 0  # Track the number of consecutive epochs with AUC not increasing
auc = 0

for epoch in range(1, 2000):
    loss, z = train()
    auc, ap = test(test_data.pos_edge_label_index, test_data.neg_edge_label_index)
    if epoch % 100 == 0:
        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    auc_values.append(auc)
    ap_values.append(ap)
    if auc >= (best_auc - 0.01 * best_auc):  # Check if auc is less than or equal to 1% decrease
        if auc >= 0.8:
            best_auc = auc
            consecutive_epochs = 0
    else:
        consecutive_epochs += 1
    # if consecutive_epochs >= 50:
    #     print('Early stopping: AUC has not increased by more than 1% for 10 epochs.')
    #     break


#####################################################################################################################

# Perform KMeans clustering on the latent space (z)

z = z.cpu().detach().numpy()
initial_centers = z[:30]
gnn_kmeans = KMeans(n_clusters=num_clusters, init=initial_centers, n_init=1, random_state=42).fit(z)
gnn_eval_data = dataset_[0]
gnn_X = gnn_eval_data.x[:, [0, 1]].cpu().detach().numpy()
gnn_df = pd.DataFrame(gnn_X, columns=['x', 'y'])

# Adding cluster labels to the DataFrame
gnn_df_with_cluster = gnn_df.copy(deep=True)
gnn_df_with_cluster['cluster'] = gnn_kmeans.labels_

gnn_G = to_networkx(gnn_eval_data)
gnn_G = gnn_G.to_undirected()
gnn_labels = gnn_kmeans.labels_
gnn_cluster_centers = gnn_kmeans.cluster_centers_

gnn_pos = dict(zip(range(gnn_X[:, 0].size), gnn_X))
node_colors = np.argmax(df.iloc[:, -30:].values, axis=1)
# Draw the Graph
fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(gnn_df_with_cluster['x'], gnn_df_with_cluster['y'], s=20, color='grey')
nx.draw_networkx_nodes(gnn_G, gnn_pos, cmap=plt.get_cmap('tab20'), node_color=node_colors, node_size=20, ax=ax)
nx.draw_networkx_edges(gnn_G, gnn_pos, edge_color='grey', ax=ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
# plt.show()

n_clusters = len(initial_centers)
n_clusters = len(initial_centers)
kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1, random_state=42).fit(z)

df['KMeans_Labels'] = kmeans.labels_
labels = kmeans.labels_

# Plot the clusters
colors = [cmap(i) for i in np.linspace(0, 1, 30)]
# for i in range(num_clusters):
#     plt.scatter(z[kmeans.labels_ == i, 0],
#                 z[kmeans.labels_ == i, 1],
#                 color=colors[i % len(colors)], alpha=0.5)
# c=colors[i % len(colours_simp)]
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, color="red")
# plt.show()

# draw the graph
fig, ax = plt.subplots(figsize=(8, 8))

# Color nodes based on the cluster they belong to
node_colors = [colors[i % len(colors)] for i in kmeans.labels_]
nx.draw_networkx_nodes(G, pos, label=kmeans.labels_, node_color=node_colors, node_size=20, ax=ax)

# Draw edges
nx.draw_networkx_edges(G, pos, ax=ax)

# Draw the remaining elements
ax.scatter(df['x'], df['y'], s=20, color='black')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.savefig(f'./data/one-hot/gnn_kmeans_results/pic/gnn_kmeans_time{run_index}.eps',format='eps', dpi=300)
plt.savefig(f'./data/one-hot/gnn_kmeans_results/pic/gnn_kmeans_time{run_index}.png',format='png', dpi=300)

plt.clf()
# Show the plot
# plt.show()

df[['rid', 'x', 'y', 'KMeans_Labels']].to_csv(f'./data/one-hot/gnn_kmeans_results/txt/gnn_kmeans_results_time{run_index}.csv', index=False)


# Good values for AUC and AP
good_auc = 0.8
good_ap = 0.5

# Plot AUC & AP values
x_values = range(1, len(auc_values) + 1)
plt.plot(x_values, auc_values, label='AUC')
plt.plot(x_values, ap_values, label='AP')
plt.axhline(y=good_auc, color='blue', linestyle='dashed', label=f'Good AUC: {good_auc}')
plt.axhline(y=good_ap, color='orange', linestyle='dashed', label=f'Good AP: {good_ap}')
plt.xlabel('Epochs/100')
plt.ylabel('Value')
plt.title('AUC and AP Progression')
plt.legend()
plt.savefig(f'./data/one-hot/gnn_kmeans_results/pic/AUC_AP_{run_index}.eps', format='eps', dpi=300)
plt.savefig(f'./data/one-hot/gnn_kmeans_results/pic/AUC_AP_{run_index}.png', format='png', dpi=300)
# plt.show()


def clear_raw_data(raw_dir):
    for filename in os.listdir(raw_dir):
        file_path = osp.join(raw_dir, filename)
        if osp.isfile(file_path):
            os.remove(file_path)


clear_raw_data(osp.join(root, 'raw'))
