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


# Reinforcement_Learning_GNN

###https://collab.dvb.bayern/display/TUMdlma/Exploiting+Edge+Features+in+Graph+Neural+Networks
Graph Neural Networks
These are networks that directly operate on the graph structure. Graph Neural Networks are typically used for the purposes of node classification, edge classification or a classification of the network as a whole. The typical Graph Neural Network learns from two matrices that are constructed from the underlying graph.

Feature Descripton Matrix (H): Aggregate of features from all nodes of the network. 



Adjacency Matrix (A): Encodes the Structure of Edges. 
A Typical layer of a Graph Neural Network can be compactly represented by the following mathematical expression:

H(l+1)=f(H(l),A)

The Adjacency Matrix is binary with 1 denoting the presence of a edge between the nodes that are given by the respective matrix indices. A 0 denotes the absence of an edge. Each node is connected to itself. The matrix is binary as the graph under consideration is non-directional and non-weighted. 

Graph Attention Network:
In order to overcome the drawbacks of the GCN, the GAT was proposed. It is a non-spectral approach in that the convolutions are performed directly on spatially close neighbours. Attention is a mechanism by which the network finds out how infuential the features of the neighbouring nodes are in learning the feature of the node under consideration. The basic building block of the GAT is the Graph Attention Layer. The following sections explain how the GAT works.

Step 1: Linear Transformation
The first step performed by the Graph Attention Layer is to apply a linear transformation - Weighted matrix W  to the feature vectors of the nodes. 

Step 2: Computation of Attention Coefficients
Attention Coefficients determine the relative importance of neighbouring features with respect to each other. They are calculated using the formula:  eij=a(Whi→,Whj→). Here a is function that we determine subject to the restriction that a:RF′×RF′→R. i and j are neighbouring nodes.
Step 3: Normalization of Attention Coefficients
Due to the varied strcuture of graphs, nodes can have a different number of neighbours. In order to have a common scaling acroos all neighbourhoods, the Attention coefficients are Normalized. 

αij=exp(LeakyReLU(eij))∑k∈Nexp(LeakyReLU(eij))
Here N is the Neighbouhood of node i.

Step 4: Computation of Final Output Features
Now we compute the learned features of nodes. σ is a Non-Linear Transformation.
h′i→=σ(∑j∈NαijWh′j→)

Step 5: Computation of Multiple Attention Mechanisms
In order to improve the stability of the learning process Multi-head attention is employed. We compute multiple different attention maps and finally aggregate all the learned representations.

h′i→=σ(1K∑k=1K∑j∈NαkijWkh′j→)
K denotes the number of independant attention maps used.





How GNN used to extract the feature of Graph Nodes

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
The graph-level representation obtained in the readout phase can be used for various tasks. It can serve as input to a downstream machine learning model for tasks such as classification, regression, or clustering

### How MPNN works ?