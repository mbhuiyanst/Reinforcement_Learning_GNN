# Reinforcement_Learning_GNN
How GNN used to extract the feature of Graph Nodes

Message Passing Neural Networks (MPNNs) are a type of neural network designed for processing graph-structured data. The basic idea behind MPNNs is to iteratively update the hidden states of nodes in a graph by passing and aggregating information between neighboring nodes. This enables the network to capture complex relationships and patterns within the graph.
Here's a step-by-step explanation of how MPNNs work for feature extraction from a graph:

1. Initialization:
Each node in the graph is assigned an initial hidden state (feature vector). This initial state could represent some known information about the node, such as node attributes.
2. Message Passing:
The network iteratively passes messages between neighboring nodes. The message 
�
�
M 
t
​
  for each node 
�
�
V 
t
​
  is computed by aggregating information from neighboring nodes. The aggregation function is typically a learned function that considers the hidden states and edges of neighboring nodes.
�
�
=
∑
�
∈
Neighbors
(
�
�
)
�
�
(
hidden state of 
�
,
edge features
)
M 
t
​
 =∑ 
v∈Neighbors(V 
t
​
 )
​
 M 
t
​
 (hidden state of v,edge features)

3. Node State Update:
The hidden state of each node is updated based on the received messages and its own current hidden state. The update function 
�
�
U 
t
​
  is a learned function that combines the previous hidden state with the aggregated message.
New Hidden State
�
=
�
�
(
previous hidden state
,
�
�
)
New Hidden State 
t
​
 =U 
t
​
 (previous hidden state,M 
t
​
 )

This process is repeated for a fixed number of message-passing steps.

4. Readout/Pooling:
After the message-passing iterations, the final step involves aggregating the hidden states of all nodes to obtain a graph-level representation. Different aggregation functions can be used, such as summation, averaging, or even more complex pooling operations.
Graph-level Representation
=
Aggregation
(
{
hidden states of all nodes
}
)
Graph-level Representation=Aggregation({hidden states of all nodes})

5. Prediction or Further Processing:
The graph-level representation obtained in the readout phase can be used for various tasks. It can serve as input to a downstream machine learning model for tasks such as classification, regression, or clustering
