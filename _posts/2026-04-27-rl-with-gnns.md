---
layout: distill
title: "Using Graph Neural Networks in Reinforcement Learning: A Practical Guide"
description: Graph Neural Networks (GNNs) have achieved excellent results for modelling relational data in many supervised learning domains. However, much fewer works have explored their potential in Reinforcement Learning (RL) despite the ubiquity of practical problems defined over graphs. In this blog post, we discuss how GNNs can be effectively integrated in Deep RL frameworks, covering crucial design decisions and practical implementation concerns. In doing so, we hope to facilitate unlocking new capabilities for RL agents to reason in graph-structured environments with dynamic action spaces and varying input sizes.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
authors:
  - name: Anonymous


# must be the exact same name as your blogpost
bibliography: 2026-04-27-rl-with-gnns.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Preliminaries
    subsections:
      - name: Reinforcement Learning
      - name: Graph Neural Networks
  - name: Limitations of Traditional Architectures for Deep Reinforcement Learning
    subsections:
      - name: Permutation Sensitivity
      - name: Fixed Output Dimensions
      - name: Bounded Input Dimensions
  - name: Designing Environments for Graph Problems
    subsections:
      - name: Fixed Action Spaces
      - name: Neighbours as Actions
      - name: "Nodes as Actions: Score-Based"
      - name: "Nodes as Actions: Proto-Action"
      - name: Edges as Actions
      - name: Invalid Action Handling
  - name: Implementation Example
    subsections:
      - name: A Note on SB3 Integration
      - name: The Actor Critic Architecture
      - name: Features Extractor
      - name: Defining the GNN
      - name: Defining the Processor Network
      - name: Defining the Actor Network
      - name: Putting Together the Complete Policy
      - name: Defining the Environment
      - name: Training the Policy
      - name: Changing Graph Size at Test Time
  - name: Future Avenues
  - name: Conclusion


---


## Introduction

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/gnn_main.svg" class="img-fluid" alt="A graph neural network processes observations in graph format and outputs to an action space with dimension depending on the input graph." %}

Graph Neural Networks (GNNs) have gained significant attention in recent years due to their ability to model relational data and capture complex interactions between entities.
To date, GNNs have mostly been applied in the supervised and unsupervised learning paradigms for tasks such as node classification, link prediction, and graph classification.

Deep reinforcement learning (RL) has also been an area of active research, with many successful applications in games, robotics, and control tasks.
However, the potential of GNNs in RL remains relatively underexplored.
Compared to traditional deep learning architectures such as convolutional neural networks (CNNs) and multi-layer perceptrons (MLPs), GNNs offer several advantages that enable novel capabilities in RL settings when used as a policy or value function approximator.
These include being agnostic to input size, permutation invariance, and the ability to handle variable action spaces.
These properties have great value in applications such as multi-agent systems, navigation, combinatorial optimisation, and resource allocation.

We hypothesise that the lack of uptake of GNNs in RL is due to unclear design patterns for integrating GNNs into RL frameworks, as well as a lack of implementation support in popular RL libraries.
Thus, in this blog post, we aim to provide a comprehensive overview of GNNs in RL, focusing on the practical design aspects of using GNNs as policy or value function approximators.
We discuss common approaches to representing environments as graphs, defining action spaces, and handling invalid actions.
Furthermore, we include a detailed implementation example using Stable Baselines 3 (SB3) <d-cite key="raffin2021stable"></d-cite> and PyTorch Geometric <d-cite key="fey2019fast"></d-cite>, two of the most widely used RL and GNN libraries respectively.

We hope that this post and [associated code](https://anonymous.4open.science/r/RL-with-GNNs-7B7E/README.md) will serve as a useful starting point for researchers and practitioners interested in leveraging GNNs in RL settings.

## Preliminaries

### Reinforcement Learning
RL is a method of solving sequential decision-making problems in the form of Markov Decision Processes (MDP).
An MDP is defined as a tuple $$\langle S, A, T, R, \gamma \rangle$$, where $$S$$ is the set of states, $$A$$ is the set of actions, $$T: S \times A \times S \rightarrow [0, 1]$$ is the transition function, $$R: S \times A \rightarrow \mathbb{R}$$ is the reward function, and $$\gamma \in [0, 1]$$ is the discount factor.

In reinforcement learning, an agent interacts with an _environment_ over a series of time steps.
At each time step $$t$$, the environment produces an _observation_ corresponding to the current state $$s_t \in S$$, and the agent selects an _action_ $$a_t \in A$$ based on its _policy_ $$\pi(a_t | s_t)$$.
The environment then transitions to a new state $$s_{t+1}$$ according to the transition function $$T$$, and the agent receives a _reward_ $$r_t = R(s_t, a_t)$$.
The agent's objective is to learn a policy that maximises the expected cumulative reward, also known as the _return_: $$\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]$$.

RL methods fall into two main categories: _value-based_ methods and _policy-based_ methods.
Value-based methods, such as Q-learning and Deep Q-Networks (DQN), focus on estimating the Q-function $$Q : S \times A \rightarrow \mathbb{R}$$, representing the expected return for taking a particular action in a given state.
Given the Q-function, a policy can be derived by selecting the action that maximises the value.
Policy-based methods, such as Policy Gradient and Proximal Policy Optimization (PPO), directly parameterise the policy $$\pi_{\theta}(a | s)$$ and optimise the parameters $$\theta$$ to maximise the expected return. An RL policy can also be trained from expert demonstrations directly using _imitation learning_ algorithms such as Behavioral Cloning (BC).
With this approach, the agent learns to mimic the behaviour of an expert by aligning its action predictions with those from state-action pairs collected from expert trajectories.

Deep neural networks including GNNs can be used as function approximators of $$Q$$ and $$\pi$$ for scaling to environments with large state and action spaces, which is loosely referred to as _Deep RL_. A typical Deep RL architecture is shown below.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/deep_rl.svg" class="img-fluid" alt="A diagram showing the flow of data in a deep reinforcement learning architecture, from environment to observation, observation encoder, value/policy network, action, and back to environment." caption="A simplified representation of a general deep RL workflow." %}

The key neural components in this architecture are the observation encoder and the value / policy network.
The observation encoder processes raw observations from the environment (e.g., images, sensor data) into a latent representation.
The value / policy network then takes this latent representation as input and outputs either the estimated value of each action (in value-based methods) or the parameters of the action distribution (in policy-based methods).

Typical deep learning architectures use CNNs for processing image-based observations and MLPs for vector-based observations.
Policy and value networks are often implemented as MLPs, which take the encoded observation as input and output action values or action probabilities.
Notably, using MLPs requires a fixed input dimension $$d$$, according to the size of the observation space or the output of the encoder, and a fixed output dimension according to the size of the action space $$|A|$$. 

GNNs are a powerful alternative architecture that can provide a host of advantages for a variety of practical RL problems. Let us review GNNs next.

<!-- Let's have a look at some popular deep RL benchmark problems.
- **Lunar Lander**: 
  - Observation space: 8-dimensional vector -- lander coordinates, velocities, and contacts.
  - Action space: $$\{0, 1, 2, 3\}$$ -- do nothing, fire left engine, fire main engine, fire right engine. -->



### Graph Neural Networks

Graphs are a widely used mathematical representation of connected systems.
A graph $$G = (V, E)$$ consists of a set of _nodes_ $$\ V$$ and a set of _edges_  $$\ E \subseteq V \times V$$.
Nodes represent entities, while edges represent connections between them.
For example, in a social network, nodes could represent individuals while edges indicate friendships between them.
In a graph, nodes and edges can have associated _feature vectors_  $$\ \mathbf{x}_{v_i}$$ and $$\mathbf{x}_{e_{i,j}}$$.
A graph can be represented using an _adjacency matrix_ $$\ adj \in \{0, 1\}^{|V| \times |V|}$$, where $$adj_{i,j} = 1$$ if there is an edge from node $$v_i$$ to node $$v_j$$, and $$0$$ otherwise.

An _embedding_ of a graph is a mapping from the graph structure and its features to a low-dimensional vector space.
Using the embedding, we can perform various downstream tasks such as node classification, link prediction, and graph classification.
_Shallow_ graph embedding methods are manually-designed approaches using local node statistics, characteristic matrices, graph kernels.
However, these methods often fail to capture complex relationships in the graph.
_Deep_ graph embedding methods aim to learn the representation by training end-to-end with task-specific supervision signals.
Graph Neural Networks (GNNs) are a class of deep learning models designed to operate on graph-structured data.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/graph_embedding2.svg" class="img-fluid" alt="On the left is an input graph with 8 nodes. On the right is a plot demonstrating the 2-D embedding of the nodes in vector space." caption="A GNN can learn an embedding of nodes in vector space." %}

GNNs rely on the _neighbourhood aggregation_ principle: the features of a node are learned by aggregating the features of its neighbours using a learnable parameterization and an activation function.
Typically, GNNs are parameter-sharing architectures, where the same set of parameters is used across all nodes and edges in the graph, akin to the convolution operation in CNNs.
In fact, GNNs can be seen as a generalisation of CNNs to arbitrary graph structures.

#### Message Passing Neural Networks

The Message Passing Neural Network (MPNN) framework <d-cite key="gilmerNeuralMessagePassing2017"></d-cite> is a useful abstraction which unifies a number of GNN architectures.
In this framework, computation occurs in a series of message-passing layers $$l \in 1, 2, \ldots, L$$, where each layer consists of two main steps: message passing and updates.
Let $$h_{v_i}^{(l)}$$ denote the embedding of node $$v_i$$ in layer $$l$$.
Typically, the initial node embeddings are set to the node features: $$h_{v_i}^{(0)} = \mathbf{x}_{v_i},\ \forall v_i \in V$$.
We define the neighbourhood $$\mathcal{N}(v_i)$$ of node $$v_i$$ to be the set of nodes that are directly connected to it. 
Then, the following operations are performed for each node $$v_i$$:

$$\mathbf{m}_{v_i}^{(l+1)} = \sum_{v_j \in \mathcal{N}(v_i)} M^{(l)}(\mathbf{h}_{v_i}^{(l)}, \mathbf{h}_{v_j}^{(l)}, \mathbf{x}_{e_{i,j}})$$

$$\mathbf{h}_{v_i}^{(l+1)} = U^{(l)}(\mathbf{h}_{v_i}^{(l)}, \mathbf{m}_{v_i}^{(l+1)})$$

Here, $$M^{(l)}$$ is the message function that computes messages from neighbouring nodes, and $$U^{(l)}$$ is the update function that updates the node embedding based on the aggregated messages.
Typically, $$M^{(l)}$$ and $$U^{(l)}$$ are parameterised by neural networks such as MLPs.
The sum operation can be replaced with other permutation-invariant operations such as mean or max aggregation.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/aggregation-illustration2.svg" class="img-fluid" alt="A diagram showing the message passing step in a GNN. A node is highlighted in red, with an unknown node embedding vector. Each of its neighbours' node embeddings are shown being passed to the highlighted node with arrows." caption="In a GNN, the embedding of a node is updated by aggregating messages from its neighbours. Reproduced with permission from <d-cite key='darvariu2024graph'></d-cite>." %}

By applying several layers of parameterised message functions and update functions, we obtain a final embedding $$\mathbf{z}_{v_i} = \mathbf{h}_{v_i}^{(L)}$$ for each node.
This embedding captures information from its $$L$$-hop neighbourhood.
In a given layer, all nodes perform the message passing and update steps simultaneously.

So far, we have only specified how node embeddings are calculated.
In order to compute a graph-level embedding, we need to apply a readout function $$\mathcal{I}$$ to the final node embeddings:

$$\mathcal{I}(\{\mathbf{h}_{v_i}^{(L)} | v_i \in V \})$$

The readout function can be manually specified to suit the task, or can be learned  <d-cite key="ying2018hierarchical"></d-cite>. 
In order to preserve permutation invariance, the readout function must also be permutation invariant.

Many popular GNN architectures can be expressed using this message-passing framework, including Graph Convolutional Networks (GCNs) <d-cite key="kipfSemiSupervisedClassificationGraph2017"></d-cite>, Graph Attention Networks (GATs) <d-cite key="velickovic2018graph"></d-cite>, and GraphSAGE <d-cite key="hamiltonInductiveRepresentationLearning2017"></d-cite>.


## Limitations of Traditional Architectures for Deep Reinforcement Learning

Even though MLPs and CNNs are widely used in Deep RL, there are a number of limitations inherent in their representational power, which we discuss below.

### Permutation Sensitivity
Graphs nominally enjoy the property of permutation invariance: regardless of the ordering of the nodes, the properties are the same, as only the *relationships* between the nodes are important.
When we write down a graph's representation using an adjacency matrix, we implicitly create an ordering of the nodes.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/permutation2.svg" class="img-fluid" alt="A four-node graph, where three nodes are connected in a triangle and the fourth is connected to the top node in the triangle. Two different adjacency matrices which both represent the graph are shown on the left and right." caption="The same graph can be represented using different adjacency matrices depending on the ordering of the nodes." %}

If we use the matrix representation of the graph as input to a neural network, we lose the property of permutation invariance.
The two adjacency matrices above are created from the same graph. Fed to an MLP, we obtain two very different outputs.
This means that in order to train our network to, say, classify graphs based on their structure, we would have to add permutations of the training data in order to ensure that it learns to correctly classify what is fundamentally the same graph.

Let's use the game of tic-tac-toe as an example. 
This game is represented by a $$3\times 3$$ grid, in which spaces can be blank, or contain an $$\texttt{X}$$ or $$\texttt{O}$$.
A simple representation of this state would be a $$3\times 3$$ matrix with each entry corresponding to the contents of the space on the board.
This kind of state representation is easily handled by an appropriately sized CNN layer or MLP after vectorisation.
An important property of the game tic-tac-toe is that the orientation of the board does not matter: we can consider game states to be the same if they are the same under rotation or reflection.

However, without external intervention, **our network does not know this**.
Without considering this kind of permutation invariance, there are **eight times** as many tic-tac-toe states that the model must learn to solve.
In a simple environment like tic-tac-toe we can easily modify the state representation to collapse symmetries and avoid this issue.
However, in general, permutation invariance is not always an easy property to engineer.
This is where GNNs can be very useful, as permutation invariance is an intrinsic property of the network, inherently collapsing equivalent state representations "for free".

#### Experiment

Let us run a simple experiment to illustrate the impact of permutation invariance.
Given a small 5-node graph, we will generate all 120 permutations of its adjacency matrix, and pass these through simple MLP, CNN, and GNN models.
These models are randomly initialised and untrained, so we do not expect any meaningful outputs.
<div class="c-page">
  <iframe src="{{ 'assets/html/2026-04-27-rl-with-gnns/permutation_model_outputs.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%"></iframe>
</div>
From the plot, we can see that the MLP and CNN models produce widely varying outputs for different permutations of the exact same graph.
Meanwhile, the GNN produces identical outputs for all permutations.
If we were to train these models, the MLP and CNN would have to learn to map all permutations of the same graph to the same output, which is a much more difficult task than simply learning the mapping for one representation of the graph.

### Fixed Output Dimensions

In traditional Deep RL settings, the shape of the action space is fixed, given by the architecture of the policy or value network.
This means that the number of possible actions an agent can take is predetermined and does not change during learning or deployment.
This can be limiting in environments where the action space is dynamic or variable, such as in navigation tasks or multi-agent systems.
In such cases, existing approaches often resort to padding the action space to a fixed size or using hierarchical action representations, which can lead to inefficiencies and suboptimal policies.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/fixed_vs_variable2.svg" class="img-fluid" alt="(Left) An agent in a room with 6 numbered doors. Each number fits in a box representing an action space with dimension 6. (Right) An agent in a room with 7 doors. The first 6 doors are numbered, and the seventh is marked with a red question mark. The 6 numbers fit in boxes representing the action space with dimension 6, but the 7th is marked with red dashed lines on the edge of the boxes." caption="A fixed action space does not accommodate flexibility in the environment." %}

For example, suppose an agent is navigating through a building with rooms connected by doors.
If we define the action space to be the set of doors in the current room, the number of possible actions can vary depending on the room.
In a traditional RL setting, we would need to pad the action space to a fixed size, which can lead to wasted capacity and difficulty in learning.
If, at test time, the agent encounters a room with more doors than seen during training, the policy may not be able to handle the additional actions.
This limitation has led to the popularity of grid-world environments, where the action space is fixed (e.g., up, down, left, right), but this comes at the cost of realism and flexibility.
Instead, by using GNNs, we can model the environment as a graph, where nodes represent rooms and edges represent doors.
Using the neighbours of the current node as possible actions allows for a dynamic action space which can adapt to the environment's structure.

### Bounded Input Dimensions

Another limitation of traditional Deep RL architectures is their inapplicability in environments of different sizes to the fixed input dimensions of the networks.
Suppose we train an MLP policy on the adjacency matrix of a graph with $$N$$ nodes.
We could, in theory, evaluate the policy on a smaller graph with $$M < N$$ nodes by padding the adjacency matrix to size $$N \times N$$.
However, if we then test the policy on a graph with $$M > N$$ nodes, the input dimensions of the MLP will not match, and the policy will be unable to process the new graph.
In a true graph structure, the number of nodes can vary, and we may not have any guarantees about the structure of the graph that would allow us to engineer a fixed-size representation.
GNNs, on the other hand, are inherently size-invariant due to their message-passing architecture.
This means that it is possible to train a GNN-based policy on small graphs and deploy it on much larger graphs without any modification to the network architecture, which can enable emergent generalisation behaviour.

  
## Designing Environments for Graph Problems
<!-- todo: highlight advantage of GNNs in each example -->

While using GNNs can offer several advantages compared to standard learning architectures in RL settings, designing the environment that the agent interacts with (e.g., action space and transition function of the MDP) also has an important role. 
In order to use GNNs in RL, we need to represent the environment as a graph.
This means defining:
1. What is a node?
2. What is an edge?
3. What node and edge features are present?
4. What is the action space, and how does it relate to the graph structure?

Generally, nodes represent entities in the environment, while edges represent relationships or interactions between those entities.
Nodes and edges can be equipped with features that describe their properties, such as weight, status, or type.

Perhaps the most important and most difficult aspect of using GNNs in RL is defining the action space.
In traditional RL, the action space is often fixed and discrete, or continuous within a certain range.
However, when using GNNs, the action space can be more complex and dynamic, depending on the graph structure.
In the following sections, we discuss several common approaches to defining action spaces in graph-based RL environments.


### Fixed Action Spaces

The most straightforward way to use a GNN in RL is to use it as a feature extractor for environments with fixed action spaces.
In this case, the GNN processes the graph-structured observation from the environment and produces a graph or node-level embedding vector.
This vector can then passed to an MLP to produce action values or probabilities.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/fixed_action_space2.svg" class="img-fluid" alt="A GNN embedding creates feature vectors for each node in a graph. These are passed to a pooling function (sum operator) to create a graph embedding. The graph embedding is fed to a MLP + softmax block, which produces an action distribution with a fixed dimension." caption="For a fixed action space, the GNN can be used as a feature extractor, with the resulting graph or node embeddings passed to an MLP to produce action values or probabilities." %}

When using a GNN as a feature extractor, there are two main approaches for obtaining the action space from the graph embedding: pooling the node embeddings to yield a graph-level embedding, or using the node embeddings directly.
If the graph embedding is pooled to a single vector, it is important to consider the pooling method used.
Common pooling methods include mean pooling, max pooling, and sum pooling.
These methods are permutation invariant, meaning that the order of the nodes does not affect the resulting graph embedding.
However, methods like summation are sensitive to the size of the graph, which can lead to issues when embedding larger or smaller graphs.
Similarly, if using node embeddings directly, care must be taken in the selection of the aggregation method to preserve embedding properties across different graph structures, for example if a larger number of neighbours are present than seen during training.

#### Examples
+ Li et al. <d-cite key="liMessageAwareGraphAttention2021"></d-cite> approach a distributed multi-robot path planning problem where agents can communicate with adjacent robots, represented by a dynamic distance-based communication graph. At each step, an agent can take an action from {up, down, left, right, idle}. Each agent observes obstacles within their field of view, which is processed by a CNN to produce node features. These features are communicated with neighbouring agents according to the graph structure, executing the message passing of the GNN in a distributed manner. To obtain the action distribution, the aggregated node embeddings are passed to an MLP followed by a softmax layer: $$f : \mathbb{R}^d \rightarrow \mathbb{R}^{5}$$, where $$d$$ is the dimension of the node embeddings. In this case, the policy is trained using imitation learning from expert demonstrations.
Ratnabala et al. <d-cite key="ratnabalaMAGNNET2023"></d-cite> use a similar architecture to implement a centralised training, decentralised execution (CTDE) approach to a heterogeneous multi-agent task allocation task using PPO.
+ Wang et al. <d-cite key="Wang2018NerveNetLS"></d-cite> approach locomotion tasks in which a body must learn to move using a variety of morphologies. The graph representation consists of joint and body nodes, with edges indicating connectivity. In a given step, each controllable node chooses an action from a continuous space using a stochastic policy $$a_i \sim \mathcal{N}(\mu_i, \sigma_i)$$. Here, $$\mu_i = f(\mathbf{z}_{v_i})$$ for an MLP $$f$$, and $$\sigma_i$$ is a learnable vector. The policy network is trained using PPO, and the authors compare using GNN and MLP architectures for the critic network.

### Neighbours as Actions

Many environments can be naturally represented as graphs where the possible actions correspond to the neighbours of a given node.
For example, in the navigation task mentioned earlier, each room could be represented as a node, with edges connecting rooms that are directly accessible from one another.
In this case, the action space is dynamic, and is represented by the neighbours of the current node.
The action space is not limited by a maximum size as in traditional RL settings, and can vary depending on the current node.
This type of action space is particularly useful in decentralised multi-agent settings, where each agent only has local information about its neighbours.

When using neighbours as actions, the typical approach is to use the GNN to produce node-level embeddings, which are then used to score the neighbours of the current node.
From these scores, an action distribution can be created, or the highest scoring neighbour can be selected directly.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/neighbours_action_space2.svg" class="img-fluid" alt="A GNN embedding creates feature vectors for the neighbours of a node of interest within a graph. These are passed to a scoring function in the form of an MLP. The scores can be used to create an action distribution using softmax." caption="Given a node of interest, the node embeddings of its neighbours can be passed through a scoring function to generate an action distribution." %}

#### Examples
+ Goeckner et al. <d-cite key="goecknerGraphNeuralNetworkbased2024"></d-cite> pose a patrolling problem in which a team of agents must overcome partial observability, distributed communications, and agent attrition. At each step, an agent chooses a node to move to from its current location $$v$$. To do this, the GNN-based embedding of each neighbour $$ \{z_u \mid u \in \mathcal{N}(v) \}$$ is passed through a scoring MLP, $$\text{SCORE}: \mathbb{R}^d \rightarrow \mathbb{R}$$. The scores of the neighbours are then passed to a selection MLP, which outputs the index of the action to take. The policies of the agents are trained using a variant of multi-agent PPO (MAPPO).
+ Pisacane et al. <d-cite key="pisacaneReinforcementLearningDiscovers2024"></d-cite> approach a decentralised graph path search problem using only local information. Each node in the graph represents an agent, and each node is assigned an attribute vector $$\mathbf{x}_{u_i} \in \mathbb{R}^d$$. Given a target node $$u_{\text{tgt}}$$, the agent at node $$u_i$$ must select one of its neighbours to forward a message $$\mathbf{m} \in \mathbb{R}^d$$ to, with the goal of reaching the target node in as few hops as possible. To choose which neighbour should receive the message, a value estimate for each neighbour is generated using an MLP $$f$$, based on the embedding of the neighbour node and the embedding of the target node: $$v(u_i, u_{\text{tgt}}) = f_v([\mathbf{z}_{u_i} \| \mathbf{z}_{u_{\text{tgt}}}])$$. An action distribution is created by passing the value estimates through a softmax layer to obtain probabilities. The policy is trained using a variant of Advantage Actor-Critic (A2C).


### Nodes as Actions: Score-Based

More generally, we can consider the entire set of nodes in the graph as possible actions.
This is particularly useful in environments where the agent can select any node in the graph as an action, such as in combinatorial optimisation problems.
Using this action space, an agent can be trained on graphs of small sizes, and learn a policy that can be evaluated on much larger graphs at test time.

Similarly to the neighbours-as-actions approach, the node embeddings produced by the GNN can be scored to produce action values or action probabilities.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/nodes_action_space2.svg" class="img-fluid" alt="A GNN embedding creates feature vectors for all nodes in a graph. These are passed to a scoring function in the form of an MLP. The scores can be used to create an action distribution using softmax." caption="The embeddings of all nodes can be passed through a scoring function to generate an action distribution." %}


#### Examples
+ Khalil et al. <d-cite key="Khalil2017LearningCO"></d-cite> approach combinatorial optimisation problems such as the travelling salesman problem (TSP) and minimum vertex cover (MVC) using Q-learning. At each step, a node is selected from the graph to be added to the solution set. The action-value estimate for each node $$v$$ in graph state $$G$$ is given by $$Q(G, v) = f([\mathbf{z}_G \| \mathbf{z}_v])$$, where $$\mathbf{z}_G$$ is the graph-level embedding obtained via pooling and $$\mathbf{z}_v$$ is the GNN embedding of node $$v$$. Here, $$f$$ is a 2-layer MLP.
+ Antonietti et al. <d-cite key="antonietti2025magnet"></d-cite> consider mesh agglomeration as a graph partitioning problem. At each step, a point is chosen to be switched from its current partition into the other. The model is implemented using four GraphSAGE layers, followed by two linear layers. The critic then uses attentional aggregation and two further linear layers to produce a value estimate, and the model is trained using A2C. Here, the authors use action masking to prevent previously selected nodes from being selected again.
+ Infantes et al. <d-cite key="infantes2024earth"></d-cite> address satellite observation scheduling using a GNN-based policy trained with PPO. In this case, the authors obtain the action logits from a concatenation of node embeddings from each layer of the GNN, passed through a linear layer to reduce them to dimension 1.

### Nodes as Actions: Proto-Action

Another method of selecting a node is to use a "proto-action": the network outputs a vector which represents the best action given the state.
Once we know what the embedding of the desired action looks like, we can choose which action to take based on those available.
The proto-action is compared to the node embeddings of the other available actions using a scoring function, from which we can then produce a probability distribution or choose an action directly.
The inspiration for this approach comes from <d-cite key="dulac-arnoldDeepReinforcementLearning2016"></d-cite>, where the authors use a similar method to select discrete actions that have a continuous representation.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/proto_action2.svg" class="img-fluid" alt="Feature vectors for all nodes are passed through a pooling layer and an action predictor to create a proto-action. The proto-action is used to compare against node embeddings in a scoring function. The scores can be used to create an action distribution using softmax." caption="A proto-action is created from the pooled node embeddings, and compared to the embedding of each node in a scoring function to create an action distribution." %}

#### Examples
+ Darvariu et al. <d-cite key="darvariuSolvingGraphbasedPublic2021"></d-cite> approach a public goods game reformulated as finding a maximal independent set in a graph. At each step, a node is selected from the graph to add to the set until no valid nodes remain. The authors create a proto-action by first summing the node embeddings and then passing it through an MLP. An action distribution is created by taking the Euclidean distance between the proto-action and each node embedding, passing these distances through a softmax layer to obtain probabilities. The policy is trained using imitation learning.
+ Trivedi et al. <d-cite key="trivediGraphOptLearningOptimization2020"></d-cite> seek to learn generative mechanisms for graph-structured data. Edges are formed by sampling two nodes from a Gaussian policy following $$\mathbf{a}^{(1)}, \mathbf{a}^{(2)} \sim \mathcal{N}(\mathbf{\mu}, \log(\mathbf{\sigma}^2))$$ given the policy $$\pi(s) = [\mathbf{\mu}, \log(\mathbf{\sigma}^2)] = g(Enc(s))$$, where $$g$$ is a 2 layer MLP and $$Enc$$ is a GNN. The policy is trained using Soft Actor-Critic (SAC) combined with Inverse Optimal Control (IOC).

<!-- TODO: \mathcal{N} redefinition conflict with neighbour notation -->

### Edges as Actions

In some problems, the actions naturally correspond to edges in the graph rather than nodes.
For example, in a network routing problem, an agent may need to select edges to route data packets through a network.

One method of selecting edges is to decompose the edge selection into a pair of node selections.
In Dai et al. <d-cite key="daiAdversarial2018"></d-cite>, edges are added or removed in order to adversarially attack a graph-level classifier. The edge selection is posed as a two-stage process: first selecting a source node then a destination node, which reduces the breadth of the action space. Nodes are selected using a similar architecture to Khalil et al. <d-cite key="Khalil2017LearningCO"></d-cite> with a separate MLP for each stage.
In fact, the nodes do not need to be selected sequentially: Trivedi et al. <d-cite key="trivediGraphOptLearningOptimization2020"></d-cite> select both nodes simultaneously by sampling from the same Gaussian policy.
While this approach is straightforward and works with existing GNN architectures, it can be less efficient, and is not necessarily optimal if edge attributes are important.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/edges_action_space2.svg" class="img-fluid" alt="Edge embeddings are created for each edge in a graph. These are passed to a scoring function in the form of an MLP. The scores can be used to create an action distribution over edges using softmax." caption="Using an embedding of the edges, a function can be applied to create an action distribution over edges in the graph." %}

Given an edge embedding, edges can be selected in a similar manner to nodes, either through scoring or proto-action methods.

There are three main ways to obtain edge embeddings from a GNN:
1. Directly compute edge embeddings using an edge-centric GNN architecture. This approach is less common, as most GNN architectures focus on node embeddings. However, some works have proposed edge-centric GNNs that can produce edge embeddings directly, such as <d-cite key="zhaoLearningPrecodingPolicy2022a"></d-cite>, <d-cite key="yuLearningCountIsomorphisms2023"></d-cite> and <d-cite key="pengLearningResourceAllocation2024"></d-cite>. With some exceptions (e.g., <d-cite key="zheng2023road"></d-cite>) they have not been used in RL.
2. Use the node embeddings to create edge embeddings by concatenating or summing the embeddings of the two nodes that form the edge. This is straightforward, but may not capture all the information about the edge itself, especially if the edge has attributes.
3. Use a line graph transformation to convert edges into nodes, allowing the GNN to produce edge-level embeddings directly. This approach has been used in works where edge attributes are more important than nodes, such as <d-cite key="jiangCensNetConvolutionEdgeNode2019"></d-cite> and <d-cite key="caiLineGraphNeural2022"></d-cite>. However, the line graph transformation generally increases the size of the graph, and can lead to some duplication of information.

### Invalid Action Handling

In many RL environments, not all actions are valid in every state.
For example, in a navigation task, an agent may not be able to move through walls or obstacles.
When using a graph-based environment, handling of invalid actions becomes particularly important, as the flexible nature of the action space can lead to a large number of invalid actions in certain states.

There are two main approaches to handling invalid actions in RL:
1. **Action Masking**: In this approach, the policy network is modified to only output probabilities for valid actions. This can be done by applying a mask to the output of the policy network, setting the probabilities of invalid actions to zero and renormalising the remaining probabilities. This ensures that the agent only selects valid actions during training and evaluation.
2. **Invalid Action Penalties**: In this approach, the agent is allowed to select any action, but receives a penalty in the reward signal if an invalid action is selected. This penalty can be a fixed negative reward or a function of the severity of the invalid action. The agent learns to avoid invalid actions through trial and error.

Invalid action penalties generally perform worse.
The choice of penalty value requires reward engineering, which can be difficult and time-consuming.
Furthermore, in graph-based environments, the set of invalid actions can be large and dynamic, making it challenging for the agent to learn to avoid them effectively.
We will demonstrate that action masking is generally a more effective approach when using graph environments in RL settings.

#### Experiment

We will run a simple experiment to compare the performance of action masking and invalid action penalties in a GNN-based RL environment.
We will use a weighted minimum vertex cover (MVC) problem, where the agent must select nodes to cover all edges in the graph while minimising the total weight of the selected nodes.
The full environment setup is described in the [Implementation Example](#implementation-example) section below.

In this setting, invalid actions correspond to selecting nodes that have already been selected.
In a given episode, the maximum number of steps that can be taken by only selecting valid actions is equal to the number of nodes in the graph.
However, if invalid actions are allowed (and penalised), the agent may select the same node multiple times, leading to indefinite episode lengths.

We will compare two agents: one using action masking to prevent invalid actions, and one using a fixed penalty of -1 for selecting an invalid action.
The reward for selecting a valid node is equal to the negative weight of the node.
For the agent using penalties, we set the maximum episode length to be the number of nodes in the graph, to prevent indefinite episodes.
Both agents are trained using PPO with the same GNN architecture (2 GraphSAGE layers) and hyperparameters.
Training is performed on random graphs with 5, 10 and 15 nodes, and validation is performed on graphs with 15 nodes.
Below we show the validation performance of both agents across 5 seeds.

<div class="c-page">
  <iframe src="{{ 'assets/html/2026-04-27-rl-with-gnns/action_masking.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%"></iframe>
</div>

From the results, we can see that the agent using penalties instead of action masking often fails to learn a good policy, whereas the action masking agent learns a policy quickly and consistently.
Even though the graphs are quite small, the penalty-based agent generally struggles to learn to avoid invalid actions, leading to poor performance.
For some of the seeds, the penalty-based agent succeeds in learning a policy with similar performance to the action masking agent, while for others the policy has no improvement, demonstrating that this method is also sensitive to initialisation.
Clearly, this is an important design decision which can have a significant impact on the performance of GNN-based RL agents.

## Implementation Example

We will illustrate how to implement a simple GNN-based policy network using PyTorch Geometric <d-cite key="fey2019fast"></d-cite>.
This will be trained using Proximal Policy Optimisation (PPO) <d-cite key="schulman2017proximalpolicyoptimizationalgorithms"></d-cite> on a weighted minimum vertex cover (MVC) problem.
We will use Stable Baselines3 (SB3) <d-cite key="raffin2021stable"></d-cite> for the RL training loop.

The MVC problem is defined on an undirected graph $$G = (V, E)$$ with node weights $$w: V \rightarrow \mathbb{R}^+$$.
The goal is to find a subset of nodes $$C \subseteq V$$ such that every edge $$ (u, v) \in E $$ has at least one endpoint in $$ C $$, while minimising the total weight of the selected nodes $$ \sum_{v \in C} w(v) $$.
We will formulate this as a sequential decision-making problem, where at each step, the agent selects a node to add to the cover set until all edges are covered.

<!-- TODO: change to real repo -->
Full code can be found in the [accompanying GitHub repository](https://anonymous.4open.science/r/RL-with-GNNs-7B7E/README.md), along with extra network examples and environment implementations.

### A Note on SB3 Integration

SB3 provides a flexible framework for implementing custom policy networks with standard RL algorithms, such as PPO and DQN.
SB3 assumes an environment that follows the OpenAI Gym interface <d-cite key="brockman2016openai"></d-cite>, which requires defining the observation and action spaces, as well as the step and reset functions.
While Gymnasium <d-cite key="towers2024gymnasium"></d-cite> supports graph-structured or sequence-structured observations, SB3 does not natively support these types of observations, and instead requires both the observation space and action space to have pre-defined fixed dimensions.

In order to work around this limitation, we will require that our environments and policies use fixed-size graphs of size `max_nodes`, padded as necessary.
This restriction allows us to use SB3's existing functionality while still leveraging the benefits of GNNs for processing graph-structured data.
While the padding introduces some overhead, the padded graphs in matrix form will immediately be transformed back into sparse graph representations to be processed by the GNN.
The `max_nodes` parameter should be chosen based on the expected size of the graphs in the environment.
However, this parameter does not inherently impose any architectural constraints on the GNN, which means it can be **changed at test time** to allow for testing on larger graphs than seen during training.

### The Actor Critic Architecture

We will base our implementation on the actor-critic architecture provided by SB3.
As a base class, we will use `MaskableActorCriticPolicy` from the `sb3_contrib` package, which allows us to apply action masks to the action distribution.
This is useful in our environment, as not all nodes will be valid actions at each step (i.e., nodes that have already been selected cannot be selected again).

The key components of the architecture are:
1. **Features extractor**: This is typically a network that processes the raw observations from the environment into a latent representation, and is shared by both the actor and critic networks. In a GNN-based architecture, this could be an encoder that maps different types of node and edge features into a common feature space. In our example, this will be a simple transformation from the matrix representation of the graph to a PyTorch Geometric `Data` object.
2. **Processor**: This network defines the main processing of the graph-structured data, which here will be shared by both the actor and critic networks. This will be a GNN that processes the graph and produces node embeddings and a graph-level embedding.
3. **Policy and value heads**: These are the final layers that produce the action distribution and value estimates, respectively. In our case, the policy head will use a [proto-action approach](#nodes-as-actions-proto-action) to select nodes, while the value head will use the graph-level embedding to estimate the value of the current state.

More detail on implementing custom policies in SB3 can be found in the [SB3 documentation](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html).

### Features Extractor

First, we will define a simple feature extractor that converts the matrix representation of the graph into a PyTorch Geometric `Data` object.
We will assume that the environment implements observations in the form of a dictionary, where the key `node_features` maps to a matrix of shape `(max_nodes, node_dim)`, the key `edge_features` maps to a matrix of shape `(max_nodes, max_nodes, node_dim)`, and the key `adjacency_matrix` maps to a binary adjacency matrix of shape `(max_nodes, max_nodes)`.
Based on these matrix representations, we will create the corresponding `Data` object, where we remove any padding from the matrices by assuming that any node with zero edges is padding.

{% highlight python %}

from gymnasium import spaces

import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from typing import Callable, Any

def matrix_features_to_batch(
    node_features: th.Tensor,
    edge_features: th.Tensor,
    adj_matrix: th.Tensor,
) -> Batch:
    """Convert the matrix features to a PyTorch Geometric Batch object.

    Args:
        node_features (th.Tensor): b x n x f_n matrix of node features
        edge_features (th.Tensor): b x n x n x f_e matrix of edge features
        adj_matrix (th.Tensor): b x n x n binary adjacency matrix

    Returns:
        Batch: PyTorch Geometric Batch object
    """

    data_list = []
    for b in range(node_features.size(0)):
        edge_index = th.nonzero(adj_matrix[b], as_tuple=False).t()
        edge_attr = edge_features[b][edge_index[0], edge_index[1]]
        has_edge = (adj_matrix[b].sum(dim=0) > 0) | (adj_matrix[b].sum(dim=1) > 0)
        node_features_b = node_features[b][has_edge]
        data = Data(
            x=node_features_b,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        data_list.append(data)
    return Batch.from_data_list(data_list)

class MatrixObservationToGraph(BaseFeaturesExtractor):
    """
    Converts matrix-based observations to graph Batch objects.

    Args:
        observation_space (spaces.Dict): The observation space.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
    ) -> None:

        features_dim = 1  # unused
        super().__init__(observation_space, features_dim=features_dim)

    def forward(self, observations) -> Batch:
        """Convert the observations to a graph Batch object."""
        node_features = observations["node_features"]
        edge_features = observations["edge_features"]
        adj_matrix = observations["adjacency_matrix"]

        batch = matrix_features_to_batch(node_features, edge_features, adj_matrix)
        return batch

{% endhighlight %}


### Defining the GNN

Now we will define a simple GNN architecture using PyTorch Geometric.
Let's suppose that each node has a set of initial features, represented as a feature vector of dimension $$d_{in}$$.
In our GNN, we will transform these into an embedding of dimension $$d$$. 
We will define the number of message-passing layers $$L$$ in the GNN.
We will use a GAT layer for this example, which processes both node and edge features.

{% highlight python %}

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GAT(nn.Module):
    def __init__(self, in_dim, embed_dim, edge_dim=None, num_layers=2, **kwargs):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, embed_dim, edge_dim=edge_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(GATv2Conv(embed_dim, embed_dim, edge_dim=edge_dim))

    def forward(self, node_fts, edge_index, edge_attr=None, **kwargs):
        x = self.conv1(node_fts, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
        return x

{% endhighlight %}


### Defining the Processor Network

With the GNN defined, we will now create a processor network that uses the GNN to produce both node embeddings and a graph-level embedding.
These will be passed to the downstream actor and critic networks respectively.
The actor network will use both the node embeddings and graph embedding to produce an action distribution, while the critic network will use only the graph embedding to produce a value estimate.

{% highlight python %}

class GraphActorCriticProcessor(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.
    It outputs a processed graph batch for the actor and a graph embedding for the critic.

    Args:
        node_dim (int): Dimension of the node feature space.
        edge_dim (int): Dimension of the edge feature space.
        embed_dim (int): Dimension of the graph embedding space.
        pooling_type (str): Pooling type to use for graph embedding computation
            (options: "max", "mean", "sum")
        network_kwargs (dict, optional): Additional arguments to pass to the graph network.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        embed_dim: int = 64,
        pooling_type: str = "max",
        network_kwargs: dict = None,
        **kwargs,
    ):
        if pooling_type not in ["max", "mean", "sum"]:
            raise ValueError(f"Unknown pooling type {pooling_type}")

        super().__init__()
        # Save output dimensions
        # This will be used by the MaskableActorCriticPolicy to create
        # the value network (the actor network will be overridden)
        self.latent_dim_vf = embed_dim
        self.latent_dim_pi = 0  # unused

        self.embed_dim = embed_dim
        self.pooling_type = pooling_type

        if network_kwargs is None:
            network_kwargs = {}

        # Create the graph processor
        self.processor = GAT(
            in_dim=node_dim,
            embed_dim=embed_dim,
            edge_dim=edge_dim,
            **network_kwargs,
        )

    def _process_graph(self, batch: Batch) -> tuple[th.Tensor, th.Tensor]:
        """Process the graph with the graph network"""

        # Process the graph with the graph network
        node_embedding = self.processor(
            node_fts=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
        )

        # Compute the graph embedding via pooling
        if self.pooling_type == "max":
            graph_embedding = global_max_pool(node_embedding, batch.batch)
        elif self.pooling_type == "mean":
            graph_embedding = global_mean_pool(node_embedding, batch.batch)
        elif self.pooling_type == "sum":
            graph_embedding = global_add_pool(node_embedding, batch.batch)

        # Return node and graph embeddings
        return node_embedding, graph_embedding

    def forward(self, batch: Batch) -> tuple[Batch, th.Tensor]:
        """
        Forward pass of the graph processor.

        Args:
            batch (Batch): A batch of graph data.
        Returns:
            processed_batch (Batch): The processed batch with updated features, to be passed to the downstream actor network.
            graph_embedding (th.Tensor): The graph embedding tensor, to be passed to the downstream critic network.
        """
        # Process the graph
        node_embedding, graph_embedding = self._process_graph(batch)

        # Turn the processed embeddings back into a Batch object
        processed_batch = Batch(
            x=node_embedding,
            edge_index=batch.edge_index,
            graph_attr=graph_embedding,  # store graph embedding in graph_attr
            batch=batch.batch,
        )

        return processed_batch, graph_embedding

    def forward_critic(self, x: Batch) -> th.Tensor:
        """Forward pass of the critic network."""
        return self.forward(x)[1]

    def forward_actor(self, x: Batch) -> Batch:
        """Forward pass of the actor network."""
        return self.forward(x)[0]

{% endhighlight %}

### Defining the Actor Network

Next, we will define the policy network that uses the GNN to produce action probabilities.
We create a proto-action from the pooled node embeddings produced by the GNN indicating the best action to take, then use a similarity function to rank the available actions.

{% highlight python %}

class ProtoActionNetwork(nn.Module):
    """
    Action network that predicts a proto-action from the graph embedding and
    uses similarity-based matching to select a node.

    Args:
        embed_dim (int): Dimension of the embedding space.
        max_nodes (int): Maximum number of nodes in the graph.
        distance_metric (str): Distance metric to use for similarity computation
            (options: "euclidean", "cosine").
        action_predictor_layers (int): Number of layers in the action predictor network.
        temp (float): Temperature parameter for the softmax.
    """

    def __init__(
        self,
        embed_dim: int,
        max_nodes: int,
        distance_metric: str = "euclidean",
        action_predictor_layers: int = 2,
        temp: float = 1.0,
    ):
        if distance_metric not in ["euclidean", "cosine"]:
            raise ValueError(f"Unknown distance metric {distance_metric}")

        super().__init__()
        self.embed_dim = embed_dim
        self.max_nodes = max_nodes
        self.distance_metric = distance_metric
        self.softmax_temp = nn.Parameter(th.tensor(temp), requires_grad=True)

        # Define the action predictor network
        # Input will be the graph embedding, output will be the proto-action
        self.action_predictor = nn.Sequential(
            *(
                [nn.Linear(self.embed_dim, self.embed_dim)]
                + [nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim)]
                * (action_predictor_layers - 1)
            )
        )

    def compute_embedding_similarities(self, embedded_acts, proto_action):
        """Compute similarities between the embedded actions and the proto-action."""

        if self.distance_metric == "euclidean":
            similarities = -th.cdist(embedded_acts, proto_action, p=2).squeeze(-1)

        elif self.distance_metric == "cosine":
            similarities = F.cosine_similarity(embedded_acts, proto_action, dim=-1)
        else:
            raise ValueError(f"unknown distance metric {self.distance_metric}")

        return similarities

    def forward(self, batch: Batch) -> th.Tensor:
        """Forward pass of the action network.
        This method uses the graph embedding to create a proto-action by passing it through an MLP.
        Action logits are then computed as similarities between the proto-action and the node embeddings.

        Args:
            batch (Batch): A batch of graph data. Required attributes are:
                - x: Node embeddings (num_nodes, embed_dim)
                - graph_attr: Graph embeddings (batch_size, embed_dim)
                - batch: Batch vector mapping each node to its respective graph in the batch (num_nodes,)

        Returns:
            th.Tensor: (b, n) matrix of similarities between the graph embedding and the node embeddings,
                where b is the batch size and n is the number of nodes.
        """

        node_embedding = batch.x
        graph_embedding = batch.graph_attr

        # Create the proto-action from the graph embedding
        proto_action = self.action_predictor(graph_embedding)

        # Compute similarities between the graph embedding and the node embeddings per batch
        similarities = th.zeros_like(batch.batch, dtype=th.float32)
        unique_batches = batch.batch.unique()
        for batch_id in unique_batches:
            batch_mask = batch.batch == batch_id
            batch_embeddings = node_embedding[batch_mask]
            batch_target = proto_action[batch_id].unsqueeze(0)  # Target for this batch
            batch_similarities = self.compute_embedding_similarities(
                batch_embeddings, batch_target
            )
            similarities[batch_mask] = batch_similarities

        # Scale similarities by temperature
        similarities = similarities / self.softmax_temp

        # Reshape similarities along the batch dimension
        similarities = to_dense_batch(
            similarities.unsqueeze(-1),
            batch.batch,
            fill_value=-1e9,  # large negative value for padding
            max_num_nodes=self.max_nodes,
        )[0]

        return similarities

{% endhighlight %}

### Putting Together the Complete Policy

Finally, we will create the complete GNN-based policy by combining the feature extractor, processor, and actor/critic networks.
Note that we have not explicitly defined the critic network. 
This is because the critic MLP is automatically created by the `MaskableGraphActorCriticPolicy` base class, according to the `latent_dim_vf` attribute defined in the processor network.
This will map the graph embedding to a scalar value estimate, and the depth can be configured via the `net_arch` parameter.

{% highlight python %}

class MaskableGraphActorCriticPolicy(MaskableActorCriticPolicy):
    """
    Custom Actor-Critic Policy with a custom feature extractor and network architecture.

    Args:
        observation_space (spaces.Dict): The observation space.
        action_space (spaces.Discrete): The action space.
        lr_schedule (Callable[[float], float]): Learning rate schedule.
        node_dim (int): Dimension of the node feature space.
        edge_dim (int): Dimension of the edge feature space.
        embed_dim (int): Dimension of the embedding space.
        pooling_type (str): Pooling type to use for graph embedding computation
            (options: "max", "mean", "sum").
        distance_metric (str): Distance metric to use for similarity computation
            (options: "euclidean", "cosine").
        temp (float): Temperature parameter for the softmax.
        network_kwargs (dict, optional): Additional arguments to pass to the graph network.
        *args: Additional arguments passed to the base class.
        **kwargs: Additional keyword arguments passed to the base class.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Callable[[float], float],
        node_dim: int,
        edge_dim: int,
        embed_dim: int = 64,
        pooling_type: str = "max",
        distance_metric: str = "euclidean",
        temp: float = 1.0,
        network_kwargs: dict = None,
        *args,
        **kwargs,
    ):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.embed_dim = embed_dim
        self.pooling_type = pooling_type
        self.distance_metric = distance_metric
        self.temp = temp
        self.network_kwargs = network_kwargs

        kwargs.setdefault("features_extractor_class", MatrixObservationToGraph)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        # override default SB3 action net to use proto-action method
        self.action_net = ProtoActionNetwork(
            embed_dim=self.embed_dim,
            max_nodes=self.action_space.n,
            distance_metric=self.distance_metric,
            temp=self.temp,
        )

    # Override the mlp extractor to use our graph processor
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GraphActorCriticProcessor(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            embed_dim=self.embed_dim,
            pooling_type=self.pooling_type,
            network_kwargs=self.network_kwargs,
        )

    # Override to save custom parameters
    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                embed_dim=self.embed_dim,
                pooling_type=self.pooling_type,
                distance_metric=self.distance_metric,
                temp=self.temp,
                network_kwargs=self.network_kwargs,
            )
        )
        return data

{% endhighlight %}


### Defining the Environment

With the policy defined, we will now create a simple Gym environment for the weighted minimum vertex cover problem.
Here, we will use a node feature vector consisting of the node weight and a binary indicator of whether the node has been selected.
We will also define an edge feature vector consisting of a binary indicator of whether the edge is covered in the current solution.
We will use a simple reward structure, where the agent receives a negative reward equal to the weight of the selected node at each step, and the episode ends when all edges are covered.

We will not provide the full implementation of the environment here, but the key components are:

1\. The action and observation space. These are defined as fixed-size spaces, with the observation space being a dictionary containing the node features, edge features, and adjacency matrix. This will be the input to the `MatrixObservationToGraph` features extractor that we defined earlier.
{% highlight python %}

        self.action_space = gym.spaces.Discrete(self.max_nodes)
        self.observation_space = gym.spaces.Dict(
            {
                # node features: is node in mvc, node weight
                "node_features": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes, 2),
                    dtype=np.float32,
                ),
                # edge features: is edge covered in mvc
                "edge_features": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes, self.max_nodes, 1),
                    dtype=np.float32,
                ),
                "adjacency_matrix": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes, self.max_nodes),
                    dtype=np.float32,
                ),
            }
        )

{% endhighlight %}
2\. The core logic. The step function will take an action (node index), and add the node to the vertex cover set if it has not already been selected. The covered edges will be updated accordingly, and the reward will be calculated based on the node weight.
{% highlight python %}

    def step(self, action):
        if self.in_mvc[action] == 1:
            reward = -1.0  # penalty for re-adding a node
            return self._get_observation(), reward, False, False, {}

        self.in_mvc[action] = 1

        # Update covered edges
        neighbours = self.graph.edge_index[1][self.graph.edge_index[0] == action]
        for neighbour in neighbours.numpy():
            self.covered_edges[action, neighbour] = 1.0
            self.covered_edges[neighbour, action] = 1.0

        done = self._all_edges_covered()
        reward = -self.graph.x[action].item()  # reward is node weight

        return self._get_observation(), reward, done, False, {}

{% endhighlight %}
3\. The observation function. This function will take the current state of the environment and return the node feature matrix and edge feature matrix as defined, including any padding.
{% highlight python %}

    def _get_observation(self):
        node_features = np.zeros((self.max_nodes, 2), dtype=np.float32)
        node_features[: self.graph.num_nodes, 0] = self.in_mvc[: self.graph.num_nodes]
        node_features[: self.graph.num_nodes, 1] = self.graph.x.numpy()

        edge_features = np.zeros((self.max_nodes, self.max_nodes, 1), dtype=np.float32)
        edge_features[:, :, 0] = self.covered_edges

        adjacency_matrix = (
            to_dense_adj(self.graph.edge_index, max_num_nodes=self.max_nodes)
            .squeeze(0)
            .numpy()
        )

        return {
            "node_features": node_features,
            "edge_features": edge_features,
            "adjacency_matrix": adjacency_matrix,
        }
{% endhighlight %}
4\. The action masks. Here, the environment will indicate which actions (nodes) are valid at each step, i.e., nodes that have not already been selected.
{% highlight python %}

    def action_masks(self):
        return (self.in_mvc == 0) & (np.arange(self.max_nodes) < self.graph.num_nodes)

{% endhighlight %}

<!-- TODO: change to real repo -->
Full code for the environment can be found in the [accompanying GitHub repository](https://anonymous.4open.science/r/RL-with-GNNs-7B7E/README.md).

### Training the Policy

With the environment and policy defined, we can now train the GNN-based policy using SB3's PPO implementation.
We will train on randomly generated graphs of size 5, 10 and 15 nodes.
We validate the policy on graphs of size 15 at regular intervals during training.
Here we train a 2-layer GAT with embedding dimension 128 on graphs with 100k PPO steps.
For comparison, we also train GCN and GraphSAGE architectures with the same parameters. 
Below we plot the validation performance of the agents over 5 seeds.

<div class="c-page">
  <iframe src="{{ 'assets/html/2026-04-27-rl-with-gnns/rewards_compare.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%"></iframe>
</div>

From these plots, we can see that the GAT and GraphSAGE architectures with default hyperparameters are able to improve their policies with more training, while the GCN architecture struggles to learn an effective policy.


### Changing Graph Size at Test Time

In order to test on graphs larger than `max_nodes`, we can simply create a new environment with a larger `max_nodes` parameter, and load the trained policy weights into a new policy instance with the same parameter.

{% highlight python %}

def change_obs_action_space(
    policy: MaskableActorCriticPolicy,
    env: VecEnv,
) -> MaskableActorCriticPolicy:
    constructor_args = policy._get_constructor_parameters()
    constructor_args["observation_space"] = env.observation_space
    constructor_args["action_space"] = env.action_space
    new_policy = policy.__class__(**constructor_args)
    new_policy.load_state_dict(policy.state_dict())
    return new_policy

{% endhighlight %}

Using this function, we can evaluate the trained policy on larger graphs with 20 nodes:

| Model | Mean Reward | Mean Episode Length |
|-------|-------------|---------------------|
| GAT | -8.02 ± 1.42 | 17.31 ± 0.80 |
| GCN | -8.60 ± 1.41 | 17.76 ± 0.97 |
| GraphSAGE | -5.82 ± 1.21 | 14.47 ± 0.97 |

We can see that the GraphSAGE model also performs well on larger out-of-distribution graphs. However, note that default hyperparameters were used for all GNN variants. Hyperparameter tuning would improve performance and may also change the rankings of the GNN variants considered here.

<!-- | Embedding Dimension | Mean Reward | Mean Episode Length |
|-------|-------------|---------------------|
| 64 | -8.14 ± 1.44 | 17.41 ± 0.99 |
| 128 | -8.02 ± 1.42 | 17.31 ± 0.80 |
| 256 | -7.92 ± 1.60 | 17.33 ± 1.07 | -->


## Future Avenues

Using GNNs as policy or value function approximators in RL unlocks many new capabilities, but there are still a number of challenges and open questions that need to be addressed.

As discussed previously, defining the action space is a key challenge when using GNNs in RL.
Most existing works use either a fixed action space or model actions as some function of nodes or edges.
Popular GNN architectures are primarily designed to produce node-level embeddings, and edge-based actions are mostly unexplored in RL settings.
At this stage, modelling more complex action spaces, such as hybrids of fixed and graph-based actions, remains an open question.

The limitations of GNN architectures themselves can also limit their effectiveness in RL settings.
At present, many GNNs operate under the assumption of homophily: that connected nodes are more likely to share similar features or labels.
GNNs have also been designed for heterogeneous graphs (e.g., <d-cite key="wang2019heterogeneous"></d-cite>), but these require a strict bipartite structure, limiting their applicability.
At present, even if an environment can be modelled as a graph, complex structures or interactions (such as distinct node types, or higher-order relationships) may create an environment that is not well-suited to existing GNN architectures.
Furthermore, many GNNs can be prone to over-smoothing, where node embeddings become indistinguishable after multiple message-passing layers <d-cite key="rusch2023survey"></d-cite>.
This makes long-range dependencies difficult to capture, and can limit the effectiveness of GNNs in environments with large or dense graphs.

Presently, there is a lack of standardised support for graph-based environments and GNN-based function approximation in popular RL libraries and frameworks.
While libraries such as PyTorch Geometric <d-cite key="fey2019fast"></d-cite> and Deep Graph Library <d-cite key="wang2019deep"></d-cite> provide implementations of various GNN architectures, integrating these with RL frameworks such as Stable Baselines3 <d-cite key="raffin2021stable"></d-cite> or RLlib <d-cite key="liang2018rllib"></d-cite> can be non-trivial, as we have demonstrated.
Improved support for graph-based RL in these libraries would facilitate further research and development in this area.
In addition, standardised benchmarks and evaluation protocols for GNN-based RL methods would help to compare different approaches and identify best practices.

## Conclusion

GNNs offer a powerful approach for function approximation in RL settings, enabling capabilities such as permutation invariance, handling variable action spaces, and applicability with dynamic input sizes.
By representing the environment as a graph, we can leverage the strengths of GNNs to tackle practical RL problems that are difficult to solve with traditional deep learning architectures.
While there are still challenges and open questions to be addressed, the integration of GNNs into RL holds great promise for advancing the field and unlocking new applications in combinatorial optimisation, multi-agent systems, and dynamic resource allocation.
Looking forward, we hope this blogpost will encourage more research exploring the application of GNNs in RL, as well as improved support for graph-based RL in popular libraries and frameworks.

