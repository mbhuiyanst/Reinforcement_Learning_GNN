##### Introduction to Graph Neural Networks (GNNs) #######

What is a Graph?

A Graph is the type of data structure that contains nodes and edges. A node can be a person, place, or thing, and the edges define the relationship between nodes. The edges can be directed and undirected based on directional dependencies. 

Graphs are excellent in dealing with complex problems with relationships and interactions. They are used in pattern recognition, social networks analysis, recommendation systems, and semantic analysis.



Graphs with NetworkX:

The code below is influenced by Daniel Holmberg's blog and I have extended further on Graph Neural Networks in Python.

1.Create networkx’s DiGraph object “H”

2.Add nodes that contain different labels, colors, and size
3.Add edges to create a relationship between two nodes. For example, “(0,1)” means that 0 has a directional dependency on 1. We will create bidirectional relationships by adding “(1,0)”
4.Extract colors and sizes in the form of lists
5.Plot the graph using networkx’s draw function

import networkx as nx
H = nx.DiGraph()

#adding nodes
H.add_nodes_from([
  (0, {"color": "blue", "size": 250}),

  (1, {"color": "yellow", "size": 400}),

  (2, {"color": "orange", "size": 150}),

  (3, {"color": "red", "size": 600}),
  (4, {"color": "red", "size": 500})


])

#adding edges
H.add_edges_from([
  (0, 1),

  (1, 2),

  (1, 0),

  (1, 3),

  (2, 3),

  (3,0),
  (4,2),
  (4,3)


])

node_colors = nx.get_node_attributes(H, "color").values()
colors = list(node_colors)
node_sizes = nx.get_node_attributes(H, "size").values()
sizes = list(node_sizes)

#Plotting Graph


nx.draw(H, with_labels=True, node_color=colors, node_size=sizes)


In the next step, we will convert the data structure from directional to an undirectional graph using the to_undirected() function. 


#converting to undirected graph
G = H.to_undirected()
nx.draw(G, with_labels=True, node_color=colors, node_size=sizes)


# Adding Weights on the directed graph
import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
H = nx.DiGraph()

# Adding nodes
H.add_nodes_from([
    (0, {"color": "blue", "size": 250}),
    (1, {"color": "yellow", "size": 400}),
    (2, {"color": "orange", "size": 150}),
    (3, {"color": "red", "size": 600}),
    (4, {"color": "red", "size": 500})
])

# Adding edges with weights
H.add_edges_from([
    (0, 1, {"weight": 1}),
    (1, 2, {"weight": 2}),
    (1, 0, {"weight": 0.5}),
    (1, 3, {"weight": 1.5}),
    (2, 3, {"weight": 0.8}),
    (3, 0, {"weight": 0.7}),
    (4, 2, {"weight": 1.2}),
    (4, 3, {"weight": 0.9})
])

# Get node attributes
node_colors = nx.get_node_attributes(H, "color").values()

# Get edge weights

edge_weights = nx.get_edge_attributes(H, "weight").values()

# Plotting Graph

pos = nx.spring_layout(H)
nx.draw(H, pos, with_labels=True, node_color=list(node_colors), node_size=700, font_size=8, font_color='black', font_weight='bold', arrowsize=20)
nx.draw_networkx_edge_labels(H, pos, edge_labels={(i, j): f"{w}" for (i, j, w) in H.edges(data='weight')})
nx.draw_networkx_edges(H, pos, width=list(edge_weights), edge_color=list(edge_weights), edge_cmap=plt.cm.Blues, edge_vmin=0, edge_vmax=max(edge_weights))

plt.show()

What is Graph Neural Network:

Graph Neural Networks
These are networks that directly operate on the graph structure. Graph Neural Networks are typically used for the purposes of node classification, edge classification or a classification of the network as a whole. The typical Graph Neural Network learns from two matrices that are constructed from the underlying graph.

Feature Descripton Matrix (H): Aggregate of features from all nodes of the network. 



Adjacency Matrix (A): Encodes the Structure of Edges. 
A Typical layer of a Graph Neural Network can be compactly represented by the following mathematical expression:

H(l+1)=f(H(l),A)

The Adjacency Matrix is binary with 1 denoting the presence of a edge between the nodes that are given by the respective matrix indices. A 0 denotes the absence of an edge. Each node is connected to itself. The matrix is binary as the graph under consideration is non-directional and non-weighted. 



Types of Graph Neural Networks:

There are several types of neural networks, and most of them have some variation of Convolutional Neural Networks. In this section, we will be learning about the most popular GNNs. 

Graph Convolutional Networks (GCNs) are similar to traditional CNNs. It learns features by inspecting neighboring nodes. GNNs aggregate node vectors, pass the result to the dense layer, and apply non-linearity using the activation function. In short, it consists of Graph convolution, linear layer, and non-learner activation function. There are two major types of GCNs: Spatial Convolutional Networks and Spectral Convolutional Networks.

Graph Auto-Encoder Networks learn graph representation using an encoder and attempt to reconstruct input graphs using a decoder. The encoder and decoders are joined by a bottleneck layer. They are commonly used in link prediction as Auto-Encoders are good at dealing with class balance. 

Recurrent Graph Neural Networks(RGNNs) learn the best diffusion pattern, and they can handle multi-relational graphs where a single node has multiple relations. This type of graph neural network uses regularizers to boost smoothness and eliminate over-parameterization. RGNNs use less computation power to produce better results. They are used in generating text, machine translation, speech recognition, generating image descriptions, video tagging, and text summarization.

Gated Graph Neural Networks (GGNNs) are better than the RGNNs in performing tasks with long-term dependencies. Gated Graph Neural Networks improve Recurrent Graph Neural Networks by adding a node, edge, and time gates on long-term dependencies. Similar to Gated Recurrent Units (GRUs), the gates are used to remember and forget information in different states. 

Application of Graph Neural Network:

Graph Classification: we use this to classify graphs into various categories. Its applications are social network analysis and text classification.

Node Classification: this task uses neighboring node labels to predict missing node labels in a graph. 

Link Prediction: predicts the link between a pair of nodes in a graph with an incomplete adjacency matrix. It is commonly used for social networks. 

Community Detection: divides nodes into various clusters based on edge structure. It learns from edge weights, and distance and graph objects similarly. 

Graph Embedding: maps graphs into vectors, preserving the relevant information on nodes, edges, and structure.

Graph Generation: learns from sample graph distribution to generate a new but similar graph structure. 

Methods associated to GNN:

1. Message passing Neural netwok(MPNN)

2. Graph Attention Neural Network(GAT)

3. Graph Convolution Network(GCN)



Message Passing Neural Networks (MPNNs) are a type of neural network designed for processing graph-structured data. This introduces a general framework for learning features on graphs based on message passing and can handle graph data containing both node and edge features.The basic idea behind MPNNs is to iteratively update the hidden states of nodes in a graph by passing and aggregating information between neighboring nodes. This enables the network to capture complex relationships and patterns within the graph.
Here's a step-by-step explanation of how MPNNs work for feature extraction from a graph:

1. Initialization:
Each node in the graph is assigned an initial hidden state (feature vector). This initial state could represent some known information about the node, such as node attributes.
2. Message Passing:
The network iteratively passes messages between neighboring nodes. The message is computed by aggregating information from neighboring nodes. The aggregation function is typically a learned function that considers the hidden states and edges of neighboring nod
3. Node State Update:
The hidden state of each node is updated based on the received messages and its own current hidden state. The update is a learned function that combines the previous hidden state with the aggregated message.

This process is repeated for a fixed number of message-passing steps.

4. Readout/Pooling:
After the message-passing iterations, the final step involves aggregating the hidden states of all nodes to obtain a graph-level representation. Different aggregation functions can be used, such as summation, averaging, or even more complex pooling operations.

5. Prediction or Further Processing:
The graph-level representation obtained in the readout phase can be used for various tasks. It can serve as input to a downstream machine learning model for tasks such as classification, regression, or clustering.


Further explanation of MPNN:
With in the graph neural network , there are several message passing layers, these are the core building blocks of GNN and the are responsible for combining node and esge information into the node embeddings.
Lets take an example of a node in a graph, the node collect the information of its direct neighbor,which means performing the message passing, What we end up with is the information about current node state  and the information about our neighbors node states and these states usually denoted wth h. Say currently we are in at time step K, then we perform an aggregation on the neighbours states to combine their information.Finally we put our current state and combined neighbors information together to get a new state or embedding in layers k+1. This message passing done by all node and therefore , we have new embedding for every node in our graph.The size of this new embeddings is the hyper parameter and depends on the graph data we use.We have to perform several times  message passing steps and this will continue until every single node in the graph knows something about others node. This knowledge stored in each of our node embeddings and contains the feature based as well as strctural information about the nodes.Eventually we can use embeddings in oder to perform predictions as they contain all the information about the graph that we need  and this is the basic idea of GNN.

Lets formulate the MPNN operation more mathmatically, the state update for a node u, is mainly perform two alreday introduced operations aggregate and update.
Aggregate use the states of all direct neighbours v of node u and aggregate them in specific way, then then update operation use the current state in time step k and combines it with the aggregated neighbor states. This formula stays the same for all variants of message passing GNN, the only thing in which they are different is how they are perform update and aggregate functions.

# Mathmatical overview:


# Message Passing Neural Networks (MPNNs)

In the context of Message Passing Neural Networks (MPNNs), the message passing update and aggregation functions play a crucial role in capturing information from neighboring nodes and updating the node states. The update function typically involves combining information from the current node and its neighbors to generate a new representation for the node.

## Message Passing Update Function

The message obtained from neighboring nodes is computed using a function \(M_t\) that depends on hidden states and edges of the neighboring nodes:
$M_t = ∑_{neighbors w} M_t(w)$




The function \(M_t{(w)}\) could involve various operations like linear transformations, non-linear activations, or even learnable parameters.

## Node State Update (Aggregation) Function

The hidden state of node \(V_t\) is updated using a function \(U_t\) that combines the previous hidden state \(H_t\) and the newly obtained message \(M_t\). A common update function is an element-wise summation or a concatenation followed by a linear transformation:

V_t = update(H_t, M_t)


Here, {update} could be defined as:

V_t = σ(W_u [H_t, M_t] + b_u)


where σ is a non-linear activation function, \(W_u\) and \(b_u\) are learnable parameters.

## Aggregation over Time (Readout) Function

After multiple iterations of message passing, a readout function aggregates information from all nodes to generate a graph-level representation. This function is denoted as {readout} and is responsible for summarizing the information gathered from individual nodes:

Final Feature Vector = readout({U_t})


Common readout functions include summation, mean, max-pooling, or even more complex operations depending on the task.

It's important to note that the exact forms of these functions may vary based on the specific MPNN architecture, and they can be customized according to the requirements of the task at hand.









###https://collab.dvb.bayern/display/TUMdlma/Exploiting+Edge+Features+in+Graph+Neural+Networks

Implementation of Multi-Agent Reinforcement Learning for Dynamic
Resource Management in 6G in-X Subnetworks.

Here we will start with in normal scenario, where we  first replicate the GA-Net game algorithm which utilize the Attetion mechanism to reason about potential relationship among subnetworks.GA-Net also utilized improved Hard attention and multi-head self attention to minimize computing complexity.

How GA-Net works?

Graph Model Representation (G):

Models potential interference relationships among subnetworks as a graph model G.
Each subnetwork receives a current observation at each time step.Then the Observations are encoded into state encodings using multilayer perceptrons (MLP) and Gate Recurrent Units (GRU).

Here Current observations, state encodings, and interference relationships are simplified as oi, si, and ei, respectively.
 Then, the state encoding ei is used to learn the interference relationships among subnetworks by GA-Net networks, resulting in a high-level state encoding, which fuses contributions from other subnetworks.Then ei is fed into GAT and Hard-Attention as input.

 Hard-Attention:
 Hard attention is employed to generate a one-hot vector as an output, indicating which subnetworks have potential interference relationships with each other.
Hard attention outcomes produces, relationships among subnetworks are simplified, and a sub-graph Gi is obtained for each subnetwork.A GRU is used in the hard-attention mechanism to determine the weights of edges, indicating whether there is an interaction between subnetworks.

 Each subnetwork
has a different degree of relevance to a specific subnetwork,
which means each edge of the graph Gi has different weights.
At the same time, the interference between subnetworks can
be affected by multiple factors and multi-head attention is applied to extract various representation from various state
feature sub-spaces.Therefore, for sub-graph Gi, a GAT layer
with multi-head attention is trained to learn the weights of
subnetworks to subnetwork i in different state feature subspaces, Obtains the joint state encoding of agent i that contains contributions from all other subnetworks.

Also, the traditional  GRU networks cannot make full use of all subnetworks’
feature information due to a reasonable and short-sighted fact
that the sequence of subnetworks plays a pivotal role in the
procedure . Therefore,  the bidirectional GRU (BiGRU)
model used , instead of traditional GRU, so that the relationship
between subnetwork i and j also depends on states of other
subnetworks.In addition, Hard -Attention cannot acheived end-to-end backpropagation gradient , therefore The Gumbel Softmax estimator is adopted to make the hard attention mechanism differentiable, enabling training through end-to-end backpropagation.

How GAT with multi-head attention works?


 Multi-head attention mechanism employed in a Graph Attention Network (GAT) for aggregating information from other subnetworks.

 Query, Key, and Value Computation :

 Query (Qm_i): Computed as the product of the query weight matrix (Wm_q) and the state encoding (ei) of the current subnetwork i, Qm_i=Wm_q . ei

 Key (Km_i): Computed as the product of the key weight matrix (Wm_k) and the state encoding (ei) of the current subnetwork i.

Km_i=Wm_k . ei

Value (Vm_i): Computed as the product of the value weight matrix (Wm_v) and the state encoding (ei) of the current subnetwork i.
Vm_i= Wm_v . ei

Query-Key Pair Aggregation:

For each subnetwork j in the subgraph Gi, a query-key pair (Qm_j, Vm_j) is received.The weight Wm_i,j is computed as the SoftMax of the dot product of the quary Qm_i and the key Km_i
scaled by the dimensionality (dk) to prevent vanishing gradients:  wm_i,j = softmax((Qm_i \K_m_j )/ dk)

Aggregated Contribution Encoding:

Using the computed weights  wm_i,j , an aggregated contribution encoding Vm_o is calculated as a weighted sum of the values Vm_j of other subnetworks in Gi.

​
 Vm_o = ∑ wm_i . Vm_j

 Concatenation of Contributions:
 Contributions from all heads are concatenated to form the final state embedding: h^\hat{i}:
 
  h^\hat{i} = ||L{m=1} \sigma(∑{j \in Gi} wm_{i,j} \cdot W_{m_s}).















