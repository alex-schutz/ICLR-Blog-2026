---
layout: distill
title: Reinforcement Learning using Graph Neural Networks
description: Graph Neural Networks (GNNs) have been widely used in various supervised learning domains, known for their size invariance and ability to model relational data. Fewer works have explored their potential in Reinforcement Learning (RL). In this blog post, we discuss how GNNs can be effectively integrated into Deep RL frameworks, unlocking new capabilities for agents to reason with dynamic action spaces and achieve high-quality zero-shot size generalisation.
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
  - name: Traditional Deep Reinforcement Learning
    subsections:
      - name: Permutation Sensitivity
      - name: Fixed Output Dimensions
      - name: Bounded Size Generalisation
  - name: Reinforcement Learning with Graph Neural Networks
  - name: Environments as Graphs
    subsections:
      - name: Fixed Action Spaces
      - name: Neighbours as Actions
      - name: Nodes as Actions
      - name: Edges as Actions
  - name: Future Avenues
  - name: Conclusion


---


## Introduction


> add picture of GNN in RL setting. use different action spaces as examples. two graphs going into GNN, different action space outputs.

Graph Neural Networks (GNNs) have gained significant attention in recent years due to their ability to model relational data and capture complex interactions between entities.
To date, most applications of GNNs have been in paradigms such as supervised and unsupervised learning, used for tasks such as node classification, link prediction, and graph classification.

Deep reinforcement learning (RL) has also been an area of active research, with many successful applications in games, robotics, and control tasks.
However, the potential of GNNs in RL remains relatively underexplored.
Compared to traditional deep learning architectures such as convolutional neural networks (CNNs) and multi-layer perceptrons (MLPs), GNNs offer several advantages that enable novel capabilities in RL settings when used as a policy or value function approximator.
These include out-of-distribution (OOD) size generalisation, permutation invariance, and the ability to handle variable action spaces.
These properties have great value in applications such as multi-agent systems, navigation, combinatorial optimisation, and resource allocation.

We hypothesise that the lack of uptake of GNNs in RL is due to unclear design patterns for integrating GNNs into RL frameworks, as well as a lack of implementation support in popular RL libraries.
Thus, in this blog post, we aim to provide a comprehensive overview of GNNs in RL, focusing on the practical design aspects of using GNNs as policy or value function approximators.


## Preliminaries

### Reinforcement Learning
RL is a method of solving a sequential decision-making problem in the form of a Markov Decision Process (MDP).
An MDP is defined as a tuple $$\langle S, A, T, R, \gamma \rangle$$, where $$S$$ is the set of states, $$A$$ is the set of actions, $$T: S \times A \times S \rightarrow [0, 1]$$ is the transition function, $$R: S \times A \rightarrow \mathbb{R}$$ is the reward function, and $$\gamma \in [0, 1)$$ is the discount factor.

In reinforcement learning, an agent interacts with an _environment_ over a series of time steps.
At each time step $$t$$, the environment produces an _observation_ corresponding to the current state $$s_t \in S$$, and the agent selects an _action_ $$a_t \in A$$ based on its _policy_ $$\pi(a_t | s_t)$$.
The environment then transitions to a new state $$s_{t+1}$$ according to the transition function $$T$$, and the agent receives a _reward_ $$r_t = R(s_t, a_t)$$.
The agent's objective is to learn a policy that maximises the expected cumulative reward, also known as the _return_: $$\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]$$.

RL methods fall into two main categories: value-based methods and policy-based methods.
Value-based methods, such as Q-learning and Deep Q-Networks (DQN), focus on estimating the Q-function $$Q : S \times A \rightarrow \mathbb{R}$$, representing the expected return for taking a particular action in a given state.
Given the Q-function, a policy can be derived by selecting the action that maximises the value.
Policy-based methods, such as Policy Gradient and Proximal Policy Optimization (PPO), directly parameterise the policy $$\pi_{\theta}(a | s)$$ and optimise the parameters $$\theta$$ to maximise the expected return.
Both approaches can be implemented using deep neural networks as function approximators, in our case, GNNs.

### Graph Neural Networks

Graphs (also confusingly called networks) are a widely used mathematical representation of connected systems.
A graph $$G = (V, E)$$ consists of a set of nodes $$V$$ and a set of edges $$E \subseteq V \times V$$.
Nodes represent entities, while edges represent connections between them.
For example, in a social network graph, nodes could represent individuals, and edges could represent friendships between them.
In a graph, nodes and edges can have associated feature vectors $$\mathbf{x}_{v_i}$$ and $$\mathbf{x}_{e_{i,j}}$$.
A graph can be represented using an adjacency matrix $$A \in \{0, 1\}^{|V| \times |V|}$$, where $$A_{i,j} = 1$$ if there is an edge from node $$v_i$$ to node $$v_j$$, and $$0$$ otherwise.

An embedding of a graph is a mapping from the graph structure and its features to a low-dimensional vector space.
Using the embedding, we can perform various downstream tasks such as node classification, link prediction, and graph classification.
Shallow graph embedding methods are manually-designed approaches using local node statistics, characteristic matrices, graph kernels.
However, these methods often fail to capture complex relationships in the graph.
Deep graph embedding methods aim to learn the representation by training end-to-end with task-specific supervision signals.
Graph Neural Networks (GNNs) are a class of deep learning models designed to operate on graph-structured data.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/karate_graph.svg" class="img-fluid" alt="" caption="TODO: Replace this image with own version or get permission from the copyright holder https://arxiv.org/abs/1403.6652." %}

GNNs rely on the neighbourhood aggregation principle: the features of a node are learned by aggregating the features of its neighbours using a learnable parameterisation and an activation function.
Typically, GNNs are parameter sharing architectures, where the same set of parameters is used across all nodes and edges in the graph, similarly to the convolution operation in CNNs.
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

> add figure of message passing step (darvariu et al 2024)

By applying several layers of parameterised message functions and update functions, each node obtains a final embedding $$\mathbf{z}_{v_i} = \mathbf{h}_{v_i}^{(L)}$$.
This embedding captures information from its $$L$$-hop neighbourhood.
In a given layer, all nodes perform the message passing and update steps simultaneously.

So far, we have only specified how node embeddings are calculated.
In order to compute a graph-level embedding, we need to apply a readout function $$\mathcal{I}$$ to the final node embeddings:

$$\mathcal{I}(\{\mathbf{h}_{v_i}^{(L)} | v_i \in V \})$$

The readout function can be manually specified to suit the task, or can be learned  <d-cite key="ying2018hierarchical"></d-cite>. 
In order to preserve permutation invariance, the readout function must also be permutation invariant.

Many popular GNN architectures can be expressed using this message-passing framework, including Graph Convolutional Networks (GCNs) <d-cite key="kipfSemiSupervisedClassificationGraph2017"></d-cite>, Graph Attention Networks (GATs) <d-cite key="velickovic2018graph"></d-cite>, and GraphSAGE <d-cite key="hamiltonInductiveRepresentationLearning2017"></d-cite>.




## Traditional Deep Reinforcement Learning

Deep RL refers to the integration of deep learning techniques with RL algorithms.
In particular, deep neural networks are used as function approximators for the value function or policy function. 
Here's what a typical deep RL architecture might look like:

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/deep_rl.svg" class="img-fluid" alt="A diagram showing the flow of data in a deep reinforcement learning architecture, from environment to observation, observation encoder, value/policy network, action, and back to environment." caption="A simplified representation of a general deep RL workflow." %}


The key neural components in this architecture are the observation encoder and the value/policy network.
The observation encoder processes raw observations from the environment (e.g., images, sensor data) into a latent representation.
The value/policy network then takes this latent representation as input and outputs either the estimated value of each action (in value-based methods) or the parameters of the action distribution (in policy-based methods).

Typical deep learning architectures use CNNs for processing image-based observations, and MLPs for vector-based observations.
Policy and value networks are often implemented as MLPs, which take the encoded observation as input and output action values or action probabilities.
Notably, these MLP-based architectures require a fixed input dimension $$d$$, according to the size of the observation space or the output of the encoder, and a fixed output dimension according to the size of the action space $$|A|$$.

<!-- Let's have a look at some popular deep RL benchmark problems.
- **Lunar Lander**: 
  - Observation space: 8-dimensional vector -- lander coordinates, velocities, and contacts.
  - Action space: $$\{0, 1, 2, 3\}$$ -- do nothing, fire left engine, fire main engine, fire right engine. -->


As powerful and well-studied as these methods are, there are a number of limitations inherent in their representational power, which we discuss below.

### Permutation Sensitivity
Graphs nominally enjoy the property of permutation invariance: regardless of the ordering of the nodes, the properties are the same, as only the *relationships* between the nodes are important.
When we write down a graph's representation using an adjacency matrix, we implicitly create an ordering of the nodes.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/permutation.svg" class="img-fluid" alt="A four-node graph, where three nodes are connected in a triangle and the fourth is connected to the top node in the triangle. Two different adjacency matrices which both represent the graph are shown on the left and right." caption="The same graph can be represented using different adjacency matrices depending on the ordering of the nodes." %}

If we use the matrix representation of the graph as input to a neural network, we lose the property of permutation invariance.
The two adjacency matrices above are created from the same graph. Fed to an MLP, we get two very different outputs.
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
This is where GNNs can be very useful, as permutation invariance is an intrinsic property of the network, inherently collapsing equivalent state representations for free.


### Fixed Output Dimensions

In traditional deep RL settings, the shape of the action space is fixed, given by the architecture of the policy or value network.
This means that the number of possible actions an agent can take is predetermined and does not change during learning or deployment.
This can be limiting in environments where the action space is dynamic or variable, such as in navigation tasks or multi-agent systems.
In such cases, existing approaches often resort to padding the action space to a fixed size or using hierarchical action representations, which can lead to inefficiencies and suboptimal policies.

> add figure showing fixed action space vs variable action space

For example, suppose an agent is navigating through a building with rooms connected by doors.
If we define the action space to be the set of doors in the current room, the number of possible actions can vary depending on the room.
In a traditional RL setting, we would need to pad the action space to a fixed size, which can lead to wasted capacity and difficulty in learning.
If, at test time, the agent encounters a room with more doors than seen during training, the policy may not be able to handle the additional actions.
This limitation has led to the popularity of grid-world environments, where the action space is fixed (e.g., up, down, left, right), but this comes at the cost of realism and flexibility.
Instead, by using GNNs, we can model the environment as a graph, where nodes represent rooms and edges represent doors.
Using the neighbours of the current node as possible actions allows for a dynamic action space which can adapt to the environment's structure.

### Bounded Size Generalisation

Another limitation of traditional deep RL architectures is their inapplicability in environments of different sizes to the fixed input and output dimensions of the networks.
Suppose we train an MLP policy on the adjacency matrix of a graph with $$N$$ nodes.
If we then test the policy on a graph with $$M > N$$ nodes, the input and output dimensions of the MLP will not match, and the policy will be unable to process the new graph.
In a true graph structure, the number of nodes can vary, and we may not have any guarantees about the structure of the graph that would allow us to engineer a fixed-size representation.
GNNs, on the other hand, are inherently size-invariant due to their message-passing architecture.
This means that it is possible to train a GNN-based policy on small graphs and deploy it on much larger graphs without any modification to the network architecture, allowing zero-shot generalisation.


## Reinforcement Learning with Graph Neural Networks

Instead of using a traditional MLP or CNN as the policy or value network in a deep RL architecture, we can use a GNN.
This allows us to leverage the advantages of GNNs, such as variable input and output dimensions, permutation invariance, and size generalisation.

In order to use a GNN in an RL setting, we first need to represent the environment as a graph.
In many cases, the environment can be naturally represented as a graph, with nodes representing entities in the environment and edges representing relationships or interactions between those entities.
Going back to our tic-tac-toe example, we can define a node to be any of the nine possible positions on the board.
We will define edges such that two nodes are connected by an edge if they are adjacent on the board.
Finally, we will define a categorical node feature $$\in \{0, 1, 2\}$$, which tells us that the node contains a blank space, $$\texttt{X}$$, or $$\texttt{O}$$ respectively.
With this graph representation of the environment, different rotations or symmetries of the board will lead to isomorphic graphs, which the GNN will inherently treat as the same.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/tictactoe.svg" class="img-fluid" alt="The game of tic-tac-toe is represented as a graph, with 9 nodes and edges connecting positions in the same horizontal, vertical, or diagonal row. An X, O or - represents the current state of each node." caption="The state in a tic-tac-toe game can be represented as a graph, collapsing equivalent states." %}

Another example where GNNs can be useful is in multi-agent collaboration tasks.
In multi-agent systems, agents often need to communicate and coordinate with each other to achieve a common goal.
In many traditional multi-agent RL settings, it is common to assume a fixed number of agent interactions, in order to maintain a fixed observation and action space.
A policy trained in this manner therefore becomes inapplicable when the number of agents changes.
In the real world, the number of agents in a system can rarely be guaranteed, due to failures, additions, or dynamic participation.
Instead, by representing the multi-agent system as a graph, where nodes represent agents and edges represent communication links between them, we can use GNNs to process the observations and actions of the agents.
This allows us to train policies that can generalise to different numbers of agents at test time, as the GNN can handle graphs with variable size and connectivity.

While GNNs offer several advantages in RL settings, it can be non-trivial to design the graph representation of the environment and define the action space and transition function of the MDP.
In the following sections, we discuss common design approaches seen in the literature for applying GNNs in RL settings.
  
## Environments as Graphs
<!-- todo: highlight advantage of GNNs in each example -->

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
In the following sections, we discuss several common approaches to defining action spaces in GNN-based RL environments.


### Fixed Action Spaces

The most straightforward way to use a GNN in RL is to use it as a feature extractor for environments with fixed action spaces.
In this case, the GNN processes the graph-structured observation from the environment and produces a graph or node-level embedding vector.
This vector is then passed to an MLP or similar network to produce action values or action probabilities.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/fixed_action_space.svg" class="img-fluid" alt="A GNN embedding creates feature vectors for each node in a graph. These are passed to a pooling function (sum operator) to create a graph embedding. The graph embedding is fed to a MLP + softmax block, which produces an action distribution with a fixed dimension." caption="For a fixed action space, the GNN can be used as a feature extractor, with the resulting graph or node embeddings passed to an MLP to produce action values or probabilities." %}

When using a GNN as a feature extractor, there are two main approaches to obtaining the action space from the graph embedding: pooling the node embeddings to get a graph-level embedding, or using the node embeddings directly.
If the graph embedding is pooled to a single vector, it is important to consider the pooling method used.
Common pooling methods include mean pooling, max pooling, and sum pooling.
These methods are permutation invariant, meaning that the order of the nodes does not affect the resulting graph embedding.
However, methods like summation are sensitive to the size of the graph, which can lead to issues when generalising to larger or smaller graphs.
Similarly, if using node embeddings directly, care must be taken in the selection of the aggregation method to preserve generalisation to different graph structures, for example if a larger number of neighbours are present than seen during training.

#### Examples
+ Li et al. <d-cite key="liMessageAwareGraphAttention2021"></d-cite> approach a distributed multi-robot path planning problem where agents can communicate with adjacent robots, represented by a dynamic distance-based communication graph. At each step, an agent can take an action from {up, down, left, right, idle}. Each agent observes obstacles within their field of view, which is processed by a CNN to produce node features. These features are communicated with neighbouring agents according to the graph structure, executing the message passing of the GNN in a distributed manner. To obtain the action distribution, the aggregated node embeddings are passed to an MLP followed by a softmax layer: $$f : \mathbb{R}^d \rightarrow \mathbb{R}^{5}$$, where $$d$$ is the dimension of the node embeddings. In this case, the policy is trained using imitation learning from expert demonstrations.
+ Wang et al. <d-cite key="Wang2018NerveNetLS"></d-cite> approach locomotion tasks in which a body must learn to move using a variety of morphologies. The graph representation consists of joint and body nodes, with edges indicating connectivity. In a given step, each controllable node chooses an action from a continuous space using a stochastic policy $$a_i \sim \mathcal{N}(\mu_i, \sigma_i)$$. Here, $$\mu_i = f(\mathbf{z}_{v_i})$$ for an MLP $$f$$, and $$\sigma_i$$ is a learnable vector. The policy network is trained using PPO, and the authors compare using GNN and MLP architectures for the critic network.
<!-- + In <d-cite key="almasanGraphNeuralNetworks2022"></d-cite>, Almasan et al. investigate optical network routing optimisation using DQN with a GNN-based Q network. For "we limit the action set to k candidate paths for each source–destination node pair." -->

### Neighbours as Actions

Many environments can be naturally represented as graphs where the possible actions correspond to the neighbours of a given node.
For example, in the navigation task mentioned earlier, each room could be represented as a node, with edges connecting rooms that are directly accessible from one another.
In this case, the action space is dynamic, and is represented by the neighbours of the current node.
The action space is not limited by a maximum size as in traditional RL settings, and can vary depending on the current node.
This type of action space is particularly useful in decentralised multi-agent settings, where each agent only has local information about its neighbours.

When using neighbours as actions, the typical approach is to use the GNN to produce node-level embeddings, which are then used to score the neighbours of the current node.
From these scores, an action distribution can be created, or the highest scoring neighbour can be selected directly.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/neighbours_action_space.svg" class="img-fluid" alt="A GNN embedding creates feature vectors for the neighbours of a node of interest within a graph. These are passed to a scoring function in the form of an MLP. The scores can be used to create an action distribution using softmax." caption="Given a node of interest, the node embeddings of its neighbours can be passed through a scoring function to generate an action distribution." %}

#### Examples
+ Goeckner et al. <d-cite key="goecknerGraphNeuralNetworkbased2024"></d-cite> pose a patrolling problem in which a team of agents must overcome partial observability, distributed communications, and agent attrition. At each step, an agent chooses a node to move to from its current location $$v$$. To do this, the GNN-based embedding of each neighbour $$ \{z_u \mid u \in \mathcal{N}(v) \}$$ is passed through a scoring MLP, $$\text{SCORE}: \mathbb{R}^d \rightarrow \mathbb{R}$$. The scores of the neighbours are then passed to a selection MLP, which outputs the index of the action to take. The policies of the agents are trained using a variant of multi-agent PPO (MAPPO).
+ Pisacane et al. <d-cite key="pisacaneReinforcementLearningDiscovers2024"></d-cite> approach a decentralised graph path search problem using only local information. Each node in the graph represents an agent, and each node is assigned an attribute vector $$\mathbf{x}_{u_i} \in \mathbb{R}^d$$. Given a target node $$u_{\text{tgt}}$$, the agent at node $$u_i$$ must select one of its neighbours to forward a message $$\mathbf{m} \in \mathbb{R}^d$$ to, with the goal of reaching the target node in as few hops as possible. To choose which neighbour should receive the message, a value estimate for each neighbour is generated using an MLP $$f$$, based on the embedding of the neighbour node and the embedding of the target node: $$v(u_i, u_{\text{tgt}}) = f_v([\mathbf{z}_{u_i} \| \mathbf{z}_{u_{\text{tgt}}}])$$. An action distribution is created by passing the value estimates through a softmax layer to get probabilities. The policy is trained using a variant of Advantage Actor-Critic (A2C).


### Nodes as Actions

More generally, we can consider the entire set of nodes in the graph as possible actions.
This is particularly useful in environments where the agent can select any node in the graph as an action, such as in combinatorial optimisation problems.
Using this action space, an agent can be trained on graphs of small sizes, and learn a policy that generalises to much larger graphs at test time.

#### Score-Based

Similarly to the neighbours-as-actions approach, the node embeddings produced by the GNN can be scored to produce action values or action probabilities.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/nodes_action_space.svg" class="img-fluid" alt="A GNN embedding creates feature vectors for all nodes in a graph. These are passed to a scoring function in the form of an MLP. The scores can be used to create an action distribution using softmax." caption="The embeddings of all nodes can be passed through a scoring function to generate an action distribution." %}


##### Example
+ Khalil et al. <d-cite key="Khalil2017LearningCO"></d-cite> approach combinatorial optimisation problems such as the travelling salesman problem (TSP) and minimum vertex cover (MVC) using Q-learning. At each step, a node is selected from the graph to be added to the solution set. The action-value estimate for each node $$v$$ in graph state $$G$$ is given by $$Q(G, v) = f([\mathbf{z}_G \| \mathbf{z}_v])$$, where $$\mathbf{z}_G$$ is the graph-level embedding obtained via pooling and $$\mathbf{z}_v$$ is the GNN embedding of node $$v$$. Here, $$f$$ is a 2-layer MLP.

#### Proto-Action

Another method of selecting a node is to use a "proto-action": the network outputs a vector which represents the best action given the state.
Once we know what the embedding of the desired action looks like, we can choose which action to take based on those available.
The proto-action gets compared to the node embeddings of the other available actions using a scoring function, from which we can then produce a probability distribution or choose an action directly.
The inspiration for this approach comes from <d-cite key="dulac-arnoldDeepReinforcementLearning2016"></d-cite>, where the authors use a similar method to select actions in a continuous action space.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/proto_action.svg" class="img-fluid" alt="Feature vectors for all nodes are passed through a pooling layer and an action predictor to create a proto-action. The proto-action is used to compare against node embeddings in a scoring function. The scores can be used to create an action distribution using softmax." caption="A proto-action is created from the pooled node embeddings, and compared to the embedding of each node in a scoring function to create an action distribution." %}

##### Examples
+ Darvariu et al. <d-cite key="darvariuSolvingGraphbasedPublic2021"></d-cite> approach a public goods game, reformulated as finding a maximal independent set in a graph. At each step, a node is selected from the graph to add to the set, until no valid nodes remain. The authors create a proto-action by first summing the node embeddings and then passing it through an MLP. An action distribution is created by taking the Euclidean distance between the proto-action and each node embedding, passing these distances through a softmax layer to get probabilities. Here the policy is trained using imitation learning.
+ Trivedi et al. <d-cite key="trivediGraphOptLearningOptimization2020"></d-cite> seek to learn generative mechanisms for graph-structured data. Edges are formed by sampling two nodes from a Gaussian policy following $$\mathbf{a}^{(1)}, \mathbf{a}^{(2)} \sim \mathcal{N}(\mathbf{\mu}, \log(\mathbf{\sigma}^2))$$ given the policy $$\pi(s) = [\mathbf{\mu}, \log(\mathbf{\sigma}^2)] = g(Enc(s))$$, where $$g$$ is a 2 layer MLP and $$Enc$$ is a GNN. The policy is trained using Soft Actor-Critic (SAC) combined with Inverse Optimal Control (IOC).

<!-- TODO: \mathcal{N} redefinition conflict with neighbour notation -->

### Edges as Actions

In some types of problems, the actions naturally correspond to edges in the graph, rather than nodes.
For example, in a network routing problem, an agent may need to select edges to route data packets through a network.

One method of selecting edges is to decompose the edge selection into a pair of node selections.
In Darvariu et al. <d-cite key="darvariuGoaldirectedGraphConstruction2021"></d-cite>, edges are to be added to a graph in order to maximise the robustness of the graph's connectivity against edge removals. The edge construction is posed as a two-stage process: first selecting a source node, then a destination node. Nodes are selected using a similar architecture to Khalil et al. <d-cite key="Khalil2017LearningCO"></d-cite>, using a separate MLP for each stage.
In fact, the nodes do not need to be selected sequentially: Trivedi et al. <d-cite key="trivediGraphOptLearningOptimization2020"></d-cite> select both nodes simultaneously by sampling from the same Gaussian policy.
While this approach is straightforward and works with existing GNN architectures, it can be less efficient, and is not necessarily optimal if edge attributes are important.

{% include figure.liquid path="assets/img/2026-04-27-rl-with-gnns/edges_action_space.svg" class="img-fluid" alt="Edge embeddings are created for each edge in a graph. These are passed to a scoring function in the form of an MLP. The scores can be used to create an action distribution over edges using softmax." caption="Using an embedding of the edges, a function can be applied to create an action distribution over edges in the graph." %}

Given an edge embedding, edges could be selected in a similar manner to nodes, either through scoring or proto-action methods.
However, most GNN architectures do not produce edge-level embeddings directly, instead prioritising node-level embeddings.
There are two main ways to obtain edge embeddings from a GNN:
1. Use the node embeddings to create edge embeddings by concatenating or summing the embeddings of the two nodes that form the edge. This is straightforward, but may not capture all the information about the edge itself, especially if the edge has attributes.
2. Use a line graph transformation to convert edges into nodes, allowing the GNN to produce edge-level embeddings directly. This approach has been used in works where edge attributes are more important than nodes, such as <d-cite key="jiangCensNetConvolutionEdgeNode2019"></d-cite> and <d-cite key="caiLineGraphNeural2022"></d-cite>. However, the line graph transformation generally increases the size of the graph, and can lead to some duplication of information.

Some works have explored edge-centric GNN architectures which directly produce edge embeddings, such as <d-cite key="zhaoLearningPrecodingPolicy2022a"></d-cite>, <d-cite key="yuLearningCountIsomorphisms2023"></d-cite> and <d-cite key="pengLearningResourceAllocation2024"></d-cite>, but to the best of our knowledge, this approach has not yet been applied in RL settings.


## Future Avenues

Using GNNs as policy or value function approximators in RL unlocks many new capabilities, but there are still a number of challenges and open questions that need to be addressed.

As discussed previously, defining the action space is a key challenge when using GNNs in RL.
Most existing works use either a fixed action space, or model actions as some function of nodes or edges.
Popular GNN architectures are primarily designed to produce node-level embeddings, with edge-based actions not so far explored in RL settings.
At this stage, modelling more complex action spaces, such as hybrids of fixed and graph-based actions, remains an open question.

The limitations of GNN architectures themselves can also limit their effectiveness in RL settings.
At present, many GNNs operate best under the assumption of homophily: that connected nodes are more likely to share similar features or labels.
GNNs have also been designed for heterophilous graphs, but these require a strict bipartite structure, limiting their applicability.
At present, even if an environment can be modelled as a graph, complex structures or interactions (such as distinct node types, or higher-order relationships) may create an environment that is not well-suited to existing GNN architectures.
Furthermore, many GNNs can be prone to over-smoothing, where node embeddings become indistinguishable after multiple message-passing layers.
This makes long-range dependencies difficult to capture, and can limit the effectiveness of GNNs in environments with large or complex graphs.

Presently, there is a lack of standardised support for graph-based environments and GNN-based policies in popular RL libraries and frameworks.
While libraries such as PyTorch Geometric <d-cite key="fey2019fast"></d-cite> and Deep Graph Library <d-cite key="wang2019deep"></d-cite> provide implementations of various GNN architectures, integrating these with RL frameworks such as Stable Baselines3 <d-cite key="raffin2021stable"></d-cite> or RLlib <d-cite key="liang2018rllib"></d-cite> can be non-trivial.
Improved support for graph-based RL in these libraries would facilitate further research and development in this area.
In addition, standardised benchmarks and evaluation protocols for GNN-based RL methods would help to compare different approaches and identify best practices.

## Conclusion

GNNs offer a powerful approach for modelling policies in RL settings, enabling capabilities such as permutation invariance, variable action spaces, and size generalisation.
By representing the environment as a graph, we can leverage the strengths of GNNs to tackle complex RL problems that are difficult to solve with traditional deep learning architectures.
While there are still challenges and open questions to be addressed, the integration of GNNs into RL holds great promise for advancing the field and unlocking new applications.
Looking forward, we hope to see more research exploring the application of GNNs in policy networks, as well as improved support for graph-based RL in popular libraries and frameworks.

