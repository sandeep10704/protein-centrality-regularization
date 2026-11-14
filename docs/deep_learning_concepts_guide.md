# Deep Learning & Graph Neural Networks: Comprehensive Concepts Guide

> **Prepared for University Course Evaluation & Viva Voce**  
> **Course:** Deep Learning (CS / AI / Data Science)  
> **Project:** Centrality-Aware Edge Regularization for Graph Attention Networks (GATv2) on Protein Graphs  

---

## 1. What is Deep Learning?
**Deep Learning (DL)** is a specialized subfield of Machine Learning based on artificial neural networks with multiple stacked layers (hence "deep"). 

Unlike classical machine learning methods—which depend on manual feature engineering (e.g., hand-crafted texture descriptors or chemical fingerprints)—deep learning architectures learn **hierarchical representations** directly from raw data via gradient-based optimization:
- **Lower layers** capture primitive local patterns (edges in images, residue identities in proteins).
- **Higher layers** synthesize complex global semantics (object categories in computer vision, tertiary folds and catalytic enzyme functions in molecular biology).

---

## 2. What are Graph Neural Networks (GNNs)?
Standard deep learning architectures assume Euclidean, regular grid-like data structures:
- **Convolutional Neural Networks (CNNs)** operate on 2D regular grids (pixels with fixed coordinates and uniform neighbors).
- **Recurrent Neural Networks (RNNs) & Transformers** operate on 1D sequential grids (words in a sentence, audio frames).

However, many critical real-world systems are **non-Euclidean**: molecular graphs, social networks, citation networks, and biological macromolecules. Graphs have:
- Variable number of neighbors per node (arbitrary degree).
- No fixed ordering of nodes (permutation invariance).
- Complex irregular topologies.

**Graph Neural Networks (GNNs)** extend deep learning to non-Euclidean domains through the **Message Passing paradigm**. For each node $v$, its representation at layer $l+1$ is updated by aggregating feature vectors from its graph neighbors $\mathcal{N}(v)$:

$$h_v^{(l+1)} = \text{UPDATE}^{(l)} \left( h_v^{(l)}, \text{AGGREGATE}^{(l)} \left( \left\{ \text{MESSAGE}^{(l)}(h_u^{(l)}, h_v^{(l)}, e_{uv}) : u \in \mathcal{N}(v) \right\} \right) \right)$$

---

## 3. Why Can Proteins Be Represented as Graphs?
A protein is a linear polymer of 20 canonical amino acid residues linked by covalent peptide bonds that folds into a complex 3D shape. 

While the primary sequence is a 1D string (like text), biological activity is entirely determined by its **3D spatial fold** (tertiary structure):
- **Nodes ($\mathcal{V}$)**: Represent individual amino acid residues (centered at the alpha carbon atom, $C_\alpha$) or Secondary Structure Elements (SSEs: $\alpha$-helices, $\beta$-sheets).
- **Node Features ($\mathbf{X}$)**: Physicochemical properties (hydropathy, formal charge, molecular weight, van der Waals volume, isoelectric point, hydrogen bond donors/acceptors, aromaticity, and one-hot identity).
- **Edges ($\mathcal{E}$)**:
  1. **Covalent Backbone Edges**: Connecting adjacent residues in the chain ($|u - v| = 1$).
  2. **Non-Covalent Spatial Contact Edges**: Formed when two residues come within a physical contact cutoff in 3D Euclidean space:
     $$d(u, v) = \|\mathbf{r}_u - \mathbf{r}_v\|_2 \le R_c \quad (R_c = 8.0\text{ \AA})$$
- **Edge Attributes ($\mathbf{E}$)**: Physical inter-residue distance, Gaussian Radial Basis Function (RBF) distance encodings, sequence separation $|u - v|$, and covalent bond flags.

Graph representation captures both local sequence continuity and non-local 3D tertiary spatial interactions simultaneously.

---

## 4. What Does Graph Convolutional Network (GCN) Do?
Proposed by **Kipf & Welling (ICLR 2017)**, GCN is the foundational spatial-spectral GNN. It performs isotropic neighborhood averaging:

$$h_v^{(l+1)} = \sigma \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}} h_u^{(l)} W^{(l)} \right)$$

Where:
- $\tilde{d}_v = 1 + \sum_{u \in \mathcal{N}(v)} 1$ is the degree with self-loops.
- $\frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}}$ is the symmetric normalization factor.
- $W^{(l)}$ is a learnable weight matrix.
- $\sigma$ is an activation function (ReLU).

**Limitation of GCN**: GCN is **isotropic**—all neighbors are weighted purely based on their static graph degrees, regardless of their feature relevance or spatial distance.

---

## 5. How Does Graph Attention Network (GAT) Work?
Proposed by **Veličković et al. (ICLR 2018)**, GAT replaces static degree normalization with **learnable attention coefficients** $\alpha_{uv}$, allowing a node to pay more attention to specific neighbors:

$$h_v^{(l+1)} = \sigma \left( \sum_{u \in \mathcal{N}(v)} \alpha_{uv} W^{(l)} h_u^{(l)} \right)$$

In standard GAT, the attention coefficient is computed as:
$$\alpha_{uv} = \frac{\exp\left( \text{LeakyReLU}(a^T [W h_u || W h_v]) \right)}{\sum_{k \in \mathcal{N}(v)} \exp\left( \text{LeakyReLU}(a^T [W h_u || W h_k]) \right)}$$

Where:
- $W$ is a shared linear projection.
- $a$ is a learnable attention weight vector.
- $||$ denotes concatenation.

---

## 6. What is Different About GATv2?
Proposed by **Brody, Alon, & Yahav (ICLR 2022)**, GATv2 resolves the fundamental flaw of original GAT: **Static Attention**.

In standard GAT:
$$e(u, v) = a^T \text{LeakyReLU}(W [h_u || h_v]) = a_1^T W h_u + a_2^T W h_v$$
Because the linear projection $W$ precedes the non-linearity, the ranking of attention coefficients is monotonic with respect to the keys and does not dynamically depend on the query node's state! GAT computes only a "static" ranking of neighbors.

**GATv2 Fixes This by Swapping the Order of Operations**:
$$e_{\text{GATv2}}(u, v) = a^T \text{LeakyReLU}\left( W [h_u || h_v || e_{uv}] \right)$$
In GATv2, the concatenation is passed through $W$ and then the non-linearity LeakyReLU *before* the inner product with $a^T$. This makes GATv2 a **strictly universal dynamic attention mechanism**, where any node can assign custom attention to any neighbor conditioned dynamically on both endpoints and edge attributes.

> **Why GATv2 is the Primary Baseline**:  
> **GATv2 is selected as the primary baseline because the proposed edge regularization mechanism is specifically designed to modify GATv2's attention-based message-passing process. Therefore, comparing the regularized model against vanilla GATv2 provides the most direct evaluation of the proposed method. GCN is retained as an additional reference architecture.**

---

## 7. What Do Attention Coefficients ($\alpha_{uv}$) Represent?
The scalar $\alpha_{uv} \in (0, 1)$ represents the **relative importance** that node $v$ assigns to the message from neighbor $u$ during message passing. 

In our protein graphs:
- High $\alpha_{uv}$ indicates that the interaction between residue $u$ and residue $v$ is critical for determining the protein's overall structural fold or catalytic reaction.
- Low $\alpha_{uv}$ indicates an incidental or peripheral contact.
- Multi-head attention ($K$ heads) allows the model to attend to different interaction types concurrently (e.g., Head 1 attends to covalent backbone neighbors, Head 2 attends to long-range tertiary salt bridges).

---

## 8. What Does Edge Dropout (DropEdge) Do?
Proposed by **Rong et al. (ICLR 2020)**, **DropEdge** is a graph data augmentation and regularization technique that randomly removes a fraction of graph edges during training:
$$m(u, v) \sim \text{Bernoulli}(1 - P_{\text{drop}})$$

### Why DropEdge is Used:
1. **Prevents Over-Fitting**: Acts like Dropout for graph topology, creating a different perturbed subgraph at each training epoch.
2. **Mitigates Over-Smoothing**: In deep GNNs (3+ layers), repeated aggregation makes all node representations converge to an indistinguishable uniform vector. Dropping edges slows down information diffusion and preserves feature diversity.

### Why Uniform DropEdge Fails on Proteins:
Uniform DropEdge treats every edge identically ($P_{\text{drop}} = 0.20$ blindly across the whole graph). On protein graphs, it:
- Breaks covalent peptide bonds ($|u - v| = 1$), fragmenting the physical polypeptide chain.
- Destroys fragile non-local tertiary contacts ($|u - v| \ge 12$), leading to tertiary collapse and reducing accuracy by $-3.67\%$.

---

## 9. What Does Graph Centrality Mean?
Graph centrality quantifies the structural importance or topological prominence of a node within a network:
- **Degree Centrality ($C_{\text{deg}}$)**: Number of direct connections a residue forms. Measures local packing density (high in the hydrophobic core).
- **Betweenness Centrality ($C_{\text{betw}}$)**: Fraction of all shortest paths passing through node $v$. Measures bottleneck residues critical for allosteric communication.
- **Closeness Centrality ($C_{\text{close}}$)**: Inverse of average shortest path distance to all other residues. Measures topological centrality in the tertiary fold.
- **Eigenvector Centrality ($C_{\text{eigen}}$)**: Measures connection to other well-connected nodes. Identifies cohesive structural motifs and secondary structure packing.

---

## 10. How Does Original AGA-Degree Work?
Developed initially for citation graphs (Cora), **Anti-Graph-Attack (AGA) Centrality Regularization** calculates node degree centrality and applies a non-linear concave barrier:

1. **Normalized Degree**:
   $$C(v) = \frac{d(v)}{\max_{w} d(w) + \epsilon}$$
2. **Edge Importance**:
   $$I(u, v) = \frac{C(u) + C(v)}{2}$$
3. **Non-Linear Concave Shield**:
   $$S(u, v) = [\text{clamp}(I(u, v), \epsilon, 1.0)]^\gamma, \quad \gamma = 0.35$$
4. **Adaptive Retention Probability**:
   $$P_{\text{drop}}(u, v) = P_{\text{base}} \cdot (1.0 - S(u, v))$$
   $$P_{\text{retain}}(u, v) = 1.0 - P_{\text{drop}}(u, v)$$

Because $x^{0.35}$ is strictly concave, moderate-to-high centrality nodes quickly reach a shield factor near $1.0$ (protected), whereas peripheral shortcuts receive a shield near $0.0$ and are aggressively dropped.

---

## 11. What Does Bio-AGA Change for Protein Graphs?
Bio-AGA adapts the mathematical shield to respect the physical laws of macromolecules:

1. **Axiom 1: Covalent Backbone Immunity**:
   $$P_{\text{drop}}(u, v) = 0.0 \quad \text{for verified covalent peptide bonds} \ (|u - v| = 1)$$
2. **Non-Local Sequence Protection Factor ($\Omega_{\text{seq}}$)**:
   Tertiary contacts separated by $\ge 12$ residues along the sequence receive an amplified shield to prevent structural fold collapse:
   $$\Omega_{\text{seq}}(u, v) = 1.0 + \alpha_{\text{seq}} \cdot \tanh\left(\frac{|u - v|}{\tau_{\text{seq}}}\right), \quad \tau_{\text{seq}} = 12, \ \alpha_{\text{seq}} = 0.25$$
3. **Spatial Distance Affinity Decay ($\Psi_{\text{dist}}$)**:
   Closer physical contacts ($3\text{--}5\text{ \AA}$) represent stronger electrostatic or hydrogen bonds than peripheral threshold contacts ($7\text{--}8\text{ \AA}$):
   $$\Psi_{\text{dist}}(u, v) = \exp\left(-\frac{d(u, v)^2}{2\sigma_d^2}\right), \quad \sigma_d = 4.0\text{ \AA}$$
4. **Multi-Centrality Generalization**: Supports Eigenvector, Closeness, and Betweenness in addition to Degree.

---

## 12. How Do Residual Connections Help?
In **Residual GATv2 (ResGATv2)**, we add a linear identity highway projection across each attention layer:
$$h^{(l+1)} = \text{GATv2Conv}(h^{(l)}, \mathcal{E}) + W_{\text{res}}^{(l)} h^{(l)}$$

### Why Residual Connections are Essential on Proteins:
1. **Preserves Residue Identity**: In noisy graphs or under edge dropout, GAT aggregation can dilute the specific chemical identity of a catalytic residue. The residual shortcut $W_{\text{res}} h^{(l)}$ directly preserves the ego-feature signal.
2. **Enables Deeper GNNs**: Prevents vanishing gradients during backpropagation across multiple message-passing hops.
3. **Empirical Impact**: Adding residual highways to Bio-AGA improved test accuracy from $50.00\%$ to **$58.00\%$** (+8.00% gain) and lowered variance under noise.

---

## 13. How Does Graph Pooling Work?
Node-level message passing produces embedding vectors for each individual residue: $h_v \in \mathbb{R}^{d}$. 

However, enzyme classification is a **graph-level task**—we need a single vector representation for the whole protein $h_{\mathcal{G}} \in \mathbb{R}^{d_{\text{graph}}}$.

We implement **Combined Mean-Max Pooling (`mean_max`)**:
$$h_{\mathcal{G}} = \left[ \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} h_v \ \Bigg|\Bigg| \ \max_{v \in \mathcal{V}} h_v \right]$$
- **Mean Pooling** captures the global background amino acid composition of the protein.
- **Max Pooling** detects the presence of sharp, localized catalytic active sites or dominant binding motifs.
- Concatenating both provides the classification MLP head with both global and local structural contexts.

---

## 14. How Do Training and Backpropagation Work?
1. **Forward Pass**:
   - A batch of protein graphs is loaded: node features $\mathbf{X}$, edge index $\mathbf{E}$, batch vector $\mathbf{b}$.
   - Edge regularization (Bio-AGA) samples a stochastic subgraph during training.
   - Message passing through 3 GATv2 layers computes contextual residue representations.
   - Global graph pooling aggregates residues into graph embeddings.
   - A 2-layer MLP classifier maps graph embeddings to class logits: $\hat{y} \in \mathbb{R}^{6}$.
2. **Loss Calculation**: Cross-Entropy Loss measures discrepancy between predicted logits and true Enzyme Commission classes.
3. **Backpropagation**:
   - Automatic differentiation calculates gradients: $\frac{\partial \mathcal{L}}{\partial \theta}$ for all network weights $\theta$.
4. **Optimizer Step**:
   - The **Adam** optimizer updates weights using adaptive first and second moment estimates of the gradients.

---

## 15. What Loss Function and Optimizer are Used?
- **Loss Function**: **Multinomial Cross-Entropy Loss**:
  $$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log\left( \frac{\exp(\hat{y}_{i, c_i})}{\sum_{j=1}^{C} \exp(\hat{y}_{i, j})} \right)$$
  Where $B$ is batch size, $C=6$ is the number of enzyme classes, and $c_i$ is the ground-truth class.
- **Optimizer**: **Adam (Adaptive Moment Estimation)**:
  - Learning rate: $\eta = 0.001$
  - Weight decay ($L_2$ regularization): $\lambda = 10^{-4}$ to prevent overfitting.
- **Early Stopping**: Monitored on validation Macro-F1 with patience = 25 epochs to prevent over-training.

---

## 16. How are Accuracy and Macro-F1 Calculated?
Given true labels $y$ and predicted labels $\hat{y}$:

### 1. Classification Accuracy
Fraction of correctly predicted protein structures:
$$\text{Accuracy} = \frac{\sum_{i=1}^{N} \mathbf{1}(y_i == \hat{y}_i)}{N}$$

### 2. Macro-Averaged F1 Score (Macro-F1)
For each enzyme class $c \in \{1, \dots, 6\}$:
$$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}, \quad \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
$$\text{F1}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
$$\text{Macro-F1} = \frac{1}{6} \sum_{c=1}^{6} \text{F1}_c$$

**Why Macro-F1 Matters in Deep Learning**: Unlike raw accuracy, Macro-F1 weights every enzyme class equally, ensuring that the model does not achieve an artificially high score by merely memorizing the majority class. On ENZYMES (which is balanced with 100 proteins per class), Macro-F1 directly assesses multi-class balance and discriminative power.
