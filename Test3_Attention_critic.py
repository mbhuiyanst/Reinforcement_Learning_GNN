import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

# Assuming AttentionCritic and Args classes are already defined as per your previous code
class AttentionCritic(nn.Module):
    def __init__(self, input_shape, args):
        super(AttentionCritic, self).__init__()

        # Encoding
        self.encoding = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.h = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.hard_bi_GRU = nn.GRU(args.rnn_hidden_dim * 2, args.rnn_hidden_dim, bidirectional=True)
        self.hard_encoding = nn.Linear(args.rnn_hidden_dim * 2, 2)

        # Soft
        self.q = nn.Linear(args.rnn_hidden_dim, args.attention_dim, bias=False)
        self.k = nn.Linear(args.rnn_hidden_dim, args.attention_dim, bias=False)
        self.v = nn.Linear(args.rnn_hidden_dim, args.attention_dim)
        self.decoding = nn.Linear(args.rnn_hidden_dim + args.attention_dim, args.n_actions)

        self.args = args
        self.input_shape = input_shape

    def forward(self, obs, hidden_state):
        size = obs.shape[0]  # batch_size * n_agents
        obs_encoding = f.relu(self.encoding(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_out = self.h(obs_encoding, h_in)

        # Hard Attention
        if self.args.hard:
            hard_weights = self.hard_attention(h_out, size)
        else:
            hard_weights = torch.ones((self.args.n_agents, size // self.args.n_agents, 1, self.args.n_agents - 1))
            if self.args.cuda:
                hard_weights = hard_weights.cuda()

        # Soft Attention
        soft_output = self.soft_attention(h_out, hard_weights, size)

        # Combining with decoding
        final_input = torch.cat([h_out, soft_output], dim=-1)
        print(final_input)

        output = self.decoding(final_input)

        return output, h_out

    def hard_attention(self, h_out, size):
        #print("Shape of h_out before reshaping:", h_out.shape)
        
        new_shape = [size // self.args.n_agents, self.args.n_agents, self.args.rnn_hidden_dim]
        h = h_out.reshape(new_shape)


        #h = h_out.reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)  
        input_hard = []
        for i in range(self.args.n_agents):
            h_i = h[:, i]  # (batch_size, rnn_hidden_dim)
            h_hard_i = []
            for j in range(self.args.n_agents):  
                if j != i:
                    h_hard_i.append(torch.cat([h_i, h[:, j]], dim=-1))
              
            h_hard_i = torch.stack(h_hard_i, dim=0)
            input_hard.append(h_hard_i)
            
        input_hard = torch.stack(input_hard, dim=-2)
            
        input_hard = input_hard.view(self.args.n_agents - 1, -1, self.args.rnn_hidden_dim * 2)
        h_hard = torch.zeros((2 * 1, size, self.args.rnn_hidden_dim))  
        if self.args.cuda:
            h_hard = h_hard.cuda()
        h_hard, _ = self.hard_bi_GRU(input_hard, h_hard)  
        h_hard = h_hard.permute(1, 0, 2)  # (batch_size * n_agents, n_agents - 1, rnn_hidden_dim * 2)
        h_hard = h_hard.reshape(-1, self.args.rnn_hidden_dim * 2)  # (batch_size * n_agents * (n_agents - 1), rnn_hidden_dim * 2)

           
        hard_weights = self.hard_encoding(h_hard)
        hard_weights = f.gumbel_softmax(hard_weights, tau=0.01)
        #print(hard_weights)
        hard_weights = hard_weights[:, 1].view(-1, self.args.n_agents, 1, self.args.n_agents - 1)
        hard_weights = hard_weights.permute(1, 0, 2, 3)
       
        return hard_weights
       
    def soft_attention(self, h_out, hard_weights, size):
        q = self.q(h_out)
        k = self.k(h_out)
        v = f.relu(self.v(h_out))

        # Assume each item in the batch is separate (not splitting into agents)
        batch_size = q.shape[0]

        x = []
        for i in range(batch_size):
            q_i = q[i].view(1, self.args.attention_dim)
            k_i = k[i].view(1, self.args.attention_dim)
            v_i = v[i].view(1, self.args.attention_dim)

            # Scaled dot-product attention
            score = torch.matmul(q_i, k_i.T)
            scaled_score = score / np.sqrt(self.args.attention_dim)
            soft_weight = f.softmax(scaled_score, dim=-1)

            x_i = v_i * soft_weight
            x.append(x_i)

        x = torch.stack(x, dim=0).reshape(batch_size, self.args.attention_dim)
        return x

#Define Parameters and Model

class Args:
    def __init__(self):
        self.rnn_hidden_dim = 128  # example value
        self.attention_dim = 64    # example value
        self.n_actions = 10        # example value
        self.n_agents = 5          # example value
        self.cuda = False         # set to True if using GPU
        self.hard = False  
        self.input_shape = 256 


# Load the CSV data
data = pd.read_csv('C:\\Reinforcement_learning\\centralized_dataset_RSSI_remove_data1.csv')

# Create a directed graph
G = nx.DiGraph()

# Define nodes and add nodes to the graph
nodes = data['SN'].tolist()
for node in nodes:
    G.add_node(node, RSSI=data.loc[data['SN'] == node, 'RSSI'].values[0])

# Connect each node to every other node
for i in range(len(nodes)):
    for j in range(len(nodes)):
        if i != j:
            G.add_edge(nodes[i], nodes[j])

# Visualize the original graph 
plt.figure(figsize=(8, 8))
nx.draw(G, with_labels=True, node_size=1000, font_size=10, node_color='lightblue', arrows=True)
plt.title("Fully Connected Directed Graph with RSSI Values")
plt.show()

# Create adjacency matrix from the graph
adjacency_matrix = nx.to_numpy_array(G)

# Initialize the AttentionCritic model
args = Args()
model = AttentionCritic(args.input_shape, args)  # Adjusted to use args.input_shape

# Process the adjacency matrix and hidden state
processed_adj_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32).view(1, -1)  # Flatten the matrix
hidden_state = torch.randn(1, args.rnn_hidden_dim)

# Pass the processed adjacency matrix through the model
output, _ = model(processed_adj_matrix, hidden_state)


#output_values = output.detach().numpy().flatten()

# Lower the threshold
threshold = 0.5  # Adjusted to a lower value for demonstration

# Process the adjacency matrix and hidden state
processed_adj_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32).view(1, -1)
hidden_state = torch.randn(1, args.rnn_hidden_dim)

# Pass the processed adjacency matrix through the model
output, new_hidden_state = model(processed_adj_matrix, hidden_state)
print("Output shape:", output.shape)
print("Output:", output)
print("New Hidden State shape:", new_hidden_state.shape)
print("New Hidden State:", new_hidden_state)

# Assuming the model's output is now a measure of node importance, 
# use it to select the top 4 nodes
importance_scores = output.detach().numpy().flatten()
top_node_indices = np.argsort(importance_scores)[-4:]  # Indices of the top 4 nodes
top_node_names = [nodes[i] for i in top_node_indices]

# Create a subgraph with the top 4 nodes
subG = G.subgraph(top_node_names)

# Visualize the subgraph
plt.figure(figsize=(8, 8))
nx.draw(subG, with_labels=True, node_size=1000, font_size=10, node_color='lightblue', arrows=True)
plt.title("Subgraph with Top 4 Nodes based on Model Output")
plt.show()


