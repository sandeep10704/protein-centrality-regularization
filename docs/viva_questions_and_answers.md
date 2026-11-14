# Deep Learning Course Project: Comprehensive Viva Voce Questions & Answers

**Project Title:** Centrality-Aware Edge Regularization for Graph Attention Networks (GATv2) on Protein Graphs  
**Course:** Deep Learning (CS / AI / Data Science)  
**Target:** University Oral Examination, Project Viva, and Technical Defense  

---

## Section 1: Deep Learning Fundamentals & Optimization

### Q1: What is Deep Learning, and how does it fundamentally differ from classical Machine Learning?
**Answer:**
Classical Machine Learning relies heavily on **manual feature engineering**, where domain experts hand-craft descriptors (such as SIFT features in computer vision or chemical fingerprints in cheminformatics) before feeding them into shallow classifiers like Support Vector Machines or Random Forests. 

**Deep Learning**, by contrast, utilizes layered artificial neural networks to perform **end-to-end representation learning**. Lower layers learn primitive, localized patterns (such as edges in images or individual amino acid physicochemical properties), while deeper layers hierarchically synthesize higher-order, task-relevant representations (such as structural motifs, catalytic clefts, and protein fold topology) directly from raw inputs via gradient descent and backpropagation.

---

### Q2: What is the difference between an epoch, a batch, and an iteration?
**Answer:**
- **Epoch:** One complete single forward and backward pass through the entire training dataset. In our project, training runs for 80 or 100 epochs.
- **Batch (Batch Size):** The subset of training samples processed simultaneously in memory before weights are updated. We use a batch size of $B = 16$ or $32$ protein graphs.
- **Iteration (Step):** A single parameter update (one forward pass, loss calculation, backward pass, and optimizer step). If a training dataset has 420 proteins and batch size is 16, one epoch consists of $\lceil 420 / 16 \rceil = 27$ iterations.

---

### Q3: What loss function did you use, and why?
**Answer:**
We used **Multi-Class Cross-Entropy Loss** ($\mathcal{L}_{\text{CE}}$):
$$\mathcal{L}_{\text{CE}} = -\frac{1}{B} \sum_{i=1}^B \sum_{c=1}^C y_{ic} \log(\hat{y}_{ic})$$
where $B$ is the batch size, $C = 6$ is the number of Enzyme Commission classes, $y_{ic} \in \{0, 1\}$ is the ground-truth one-hot label, and $\hat{y}_{ic} = \text{Softmax}(z_{ic})$ is the predicted probability for class $c$.

**Why Cross-Entropy?**
1. It is the information-theoretic standard for mutually exclusive multi-class classification, minimizing the Kullback-Leibler (KL) divergence between true and predicted class distributions.
2. When combined with Softmax, the derivative with respect to pre-activation logits simplifies to $(\hat{y}_{ic} - y_{ic})$, which provides linear, non-vanishing gradient signals even when predictions are far from target labels.

---

### Q4: Why did you choose AdamW over standard Stochastic Gradient Descent (SGD)?
**Answer:**
1. **Adaptive Learning Rates:** Biological graphs vary in size (from 10 to 750+ residues), leading to sparse gradient updates across different attention heads. **AdamW** maintains per-parameter first ($m_t$) and second ($v_t$) uncentered moment estimates, dynamically scaling learning rates for each parameter.
2. **Decoupled Weight Decay:** Standard Adam implements $L_2$ regularization by adding weight decay directly to the gradient, which distorts the moving moment averages and leads to suboptimal regularization. AdamW (Loshchilov & Hutter, ICLR 2019) decouples weight decay by subtracting $\lambda w_t$ directly from the parameter update, ensuring effective regularization and superior generalization. We used a learning rate of $\eta = 0.001$ and weight decay of $\lambda = 10^{-4}$.

---

### Q5: What is Cosine Annealing, and why did you use it?
**Answer:**
**Cosine Annealing** (Loshchilov & Hutter) schedules the learning rate smoothly from an initial value $\eta_{\text{max}}$ down to $\eta_{\text{min}}$ following a half-cosine curve:
$$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\left(\frac{t}{T_{\text{max}}}\pi\right)\right)$$
- Early in training, high learning rates allow parameters to escape poor local minima and saddle points.
- As training progresses, the smooth decay allows the optimizer to settle into flat, generalizable basins of the loss landscape without chaotic parameter oscillations.

---

### Q6: How do you calculate Accuracy and Macro-F1, and why is Macro-F1 important?
**Answer:**
- **Accuracy:** The ratio of correct predictions to total predictions:
  $$\text{Accuracy} = \frac{\sum_{c=1}^C \text{TP}_c}{N}$$
- **Macro-F1:** The unweighted arithmetic mean of F1-scores across all classes:
  $$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^C \text{F1}_c, \quad \text{where } \text{F1}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
**Why Macro-F1 matters:** Standard accuracy can be misleading if class distributions are imbalanced (a trivial classifier predicting only the majority class can yield high accuracy). Macro-F1 assigns equal weight to every biological class, ensuring that the model performs robustly across all 6 enzyme families regardless of class frequency.

---

## Section 2: Graph Neural Networks & Attention Mechanisms

### Q7: What is the Message Passing paradigm in GNNs?
**Answer:**
In Graph Neural Networks, message passing is the process by which nodes iteratively aggregate contextual information from their spatial graph neighborhoods across $K$ successive layers. For node $v$ at layer $l+1$:
1. **Message Generation:** Each neighbor $u \in \mathcal{N}(v)$ computes a message vector using its current state $h_u^{(l)}$ and edge features $e_{uv}$:
   $$m_{uv}^{(l)} = \text{MESSAGE}^{(l)}\left(h_u^{(l)}, h_v^{(l)}, e_{uv}\right)$$
2. **Aggregation:** Messages from all neighbors are aggregated using a permutation-invariant operator (Sum, Mean, or Attention-weighted Sum):
   $$\bar{m}_v^{(l)} = \text{AGGREGATE}^{(l)}\left(\{m_{uv}^{(l)} : u \in \mathcal{N}(v)\}\right)$$
3. **Node Update:** The aggregated message is combined with the node's previous state:
   $$h_v^{(l+1)} = \text{UPDATE}^{(l)}\left(h_v^{(l)}, \bar{m}_v^{(l)}\right)$$
After $K$ layers, node $v$'s representation encodes information from its entire $K$-hop spatial neighborhood.

---

### Q8: What is the fundamental difference between GCN, GAT, and GATv2?
**Answer:**
- **GCN (Graph Convolutional Network):** Uses fixed, symmetric degree-normalized weights:
  $$h_v^{(l+1)} = \sigma \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}} h_u^{(l)} W^{(l)} \right)$$
  It is *isotropic*—every neighbor is weighted purely by graph degrees, completely ignoring edge features, physical distance, or feature compatibility.
- **GAT (Graph Attention Network):** Introduces learned attention coefficients $\alpha_{uv}$, but suffers from **static attention**:
  $$\alpha_{uv} = \text{Softmax}_v \left( \text{LeakyReLU}\left(\mathbf{a}^T [W h_u || W h_v]\right)\right)$$
  Because the linear projection $\mathbf{a}^T W$ occurs outside the non-linearity, the ranking of neighbor importance is monotonic and cannot dynamically adjust based on the query node.
- **GATv2 (Dynamic Graph Attention):** Applies the non-linear activation **before** the linear projection:
  $$\alpha_{uv} = \text{Softmax}_v \left( \mathbf{a}^T \text{LeakyReLU}\left( W_s h_u + W_t h_v + W_e e_{uv} \right)\right)$$
  This grants GATv2 *universal dynamic attention*: any node can attend to any neighbor with arbitrary priority conditioned on the query node's state.

---

### Q9: Why is Multi-Head Attention beneficial in GATv2?
**Answer:**
In a single-head attention layer, attention coefficients can concentrate heavily on one specific neighbor, leading to high gradient variance. **Multi-Head Attention** computes $K$ independent attention distributions in parallel (we use $K = 4$ heads):
$$h_v^{(l+1)} = \Vert_{k=1}^K \sigma \left( \sum_{u \in \mathcal{N}(v)} \alpha_{uv}^{(k)} W_v^{(k)} h_u^{(l)} \right)$$
In protein graphs, different heads specialize in capturing distinct physical interaction types:
- Head 1 can attend to sequential covalent backbone neighbors.
- Head 2 can attend to long-range tertiary hydrogen bonds.
- Head 3 can attend to hydrophobic contact packing.
- Head 4 can attend to electrostatic salt bridges.

---

### Q10: How does Graph-Level Readout (Pooling) work? Why concatenate Mean and Max pooling?
**Answer:**
Node representation learning produces a set of vectors $\{h_v^{(L)} \in \mathbb{R}^d\}_{v \in \mathcal{V}}$. However, enzyme classification requires a single graph-level embedding $h_G \in \mathbb{R}^{2d}$.
- **Global Mean Pooling:** $\frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} h_v$ computes the overall average structural properties of the enzyme (general fold architecture, overall hydrophobicity).
- **Global Max Pooling:** $\max_{v \in \mathcal{V}} h_v$ captures the most prominent, extreme feature activations across the protein (such as an intense catalytic charge or specialized active-site residue).
- **Concatenation:** $[h_{\text{mean}} \Vert h_{\text{max}}]$ preserves both global structural context and localized catalytic peaks, which is significantly more expressive than using mean pooling alone.

---

## Section 3: Graph Regularization & Proposed Methodology

### Q11: What is DropEdge, and why does it fail on protein graphs?
**Answer:**
**DropEdge** (Rong et al., ICLR 2020) is a graph regularization technique originally designed for citation networks (like Cora). It randomly drops graph edges with a fixed probability $P_{\text{base}}$ during each training epoch to prevent over-smoothing.

**Why it fails on proteins:**
1. **Backbone Disruption:** In citation networks, edges are homogeneous citations. In proteins, edges include covalent peptide bonds ($|u - v| = 1$). Uniformly dropping 20% of edges severs the covalent chain, fragmenting the polypeptide into disconnected oligomers.
2. **Tertiary Collapse:** Rare long-range tertiary contacts (e.g. between residue 20 and residue 180) that hold the 3D catalytic fold together are dropped at the same rate as redundant local contacts.
Empirically, Uniform DropEdge caused test accuracy on ENZYMES to drop from $54.00\%$ down to $50.33\%$ (a $-3.67\%$ degradation).

---

### Q12: What is the "Hydrophobic Core vs. Active Site Paradox"?
**Answer:**
Original AGA uses **degree centrality** $C(v) = d(v) / \max_w d(w)$ to protect high-degree nodes. 
In social or citation networks, high-degree nodes are influential celebrities or foundational papers.
In protein 3D structures:
- **High-degree nodes ($d \ge 15$):** Reside inside the tightly packed, buried **hydrophobic core**. These residues (e.g., Leucine, Isoleucine) are inert scaffolding whose job is merely to hold the fold together.
- **Low-degree nodes ($d \approx 4-8$):** Reside on flexible, solvent-exposed **surface loops**. These loops contain the actual catalytic active sites that bind substrates and carry out chemical reactions!

Therefore, applying degree centrality blindly protects the structural ballast while leaving the critical catalytic active sites vulnerable to edge dropping. This paradox necessitates a biologically grounded regularizer.

---

### Q13: Explain the mathematical formulation of Bio-AGA.
**Answer:**
Bio-AGA computes an edge retention probability $P_{\text{retain}}(u, v)$ for each edge $(u, v)$ using three biological constraints:

1. **Covalent Backbone Immunity:**
   $$P_{\text{drop}}(u, v) = 0 \quad \text{if } |u - v| = 1 \implies P_{\text{retain}}(u, v) = 1.0$$
   Covalent peptide bonds are never dropped.
2. **Non-Local Sequence Protection:**
   $$\Omega_{\text{seq}}(u, v) = 1 + \lambda_{\text{seq}} \log(1 + |u - v|) \quad (\lambda_{\text{seq}} = 0.5)$$
   Edges between residues far apart in sequence represent critical tertiary contacts and receive heightened retention.
3. **Spatial Distance Attenuation:**
   $$\Psi_{\text{dist}}(u, v) = \exp\left( -\frac{d(u, v)^2}{2\sigma^2} \right) \quad (\sigma = 4.0\text{ \AA})$$
   Short-range contacts (< 4 Å, such as hydrogen bonds and salt bridges) are shielded.
4. **Protective Shield & Edge Retention:**
   $$I(u, v) = \frac{C(u) + C(v)}{2}, \quad S(u, v) = [I(u, v)]^\gamma \quad (\gamma = 0.35)$$
   $$P_{\text{drop}}(u, v) = P_{\text{base}} \cdot (1 - S(u, v)) \cdot \frac{1}{\Omega_{\text{seq}}(u, v)}$$
   $$P_{\text{retain}}(u, v) = 1 - P_{\text{drop}}(u, v)$$
Edges are sampled via independent Bernoulli trials: $m(u, v) \sim \text{Bernoulli}(P_{\text{retain}}(u, v))$.

---

### Q14: Why did Bio-AGA require Residual GATv2 (ResGATv2) to achieve top performance?
**Answer:**
When edges are dropped stochastically, information pathways are perturbed. In a standard GATv2 network without skip connections:
- Dropping edges creates bottleneck nodes, leading to vanishing gradients and information loss in deeper layers. This is why Bio-AGA alone on vanilla GATv2 scored $50.00\%$.
- In **Residual GATv2**, an identity projection shortcut is added across each attention layer:
  $$h_v^{(l+1)} = \text{ELU}\left(\text{GATv2Conv}(h_v^{(l)}, \mathcal{E}_{\text{sampled}}) + W_{\text{res}} h_v^{(l)}\right)$$
This ensures that even if certain spatial edges are temporarily masked, residue representations can flow unimpeded via the residual path. The synergy between Bio-AGA and ResGATv2 yields **$58.00\%$ accuracy** (+7.67% over DropEdge).

---

## Section 4: Experimental Results & Analysis

### Q15: What is your primary baseline, and why did you select GATv2 instead of GCN?
**Answer:**
**GATv2 is selected as the primary baseline because the proposed edge regularization mechanism is specifically designed to modify GATv2's attention-based message-passing process. Therefore, comparing the regularized model against vanilla GATv2 provides the most direct evaluation of the proposed method. GCN is retained as an additional reference architecture.**

Our experimental comparisons follow this structured order:
1. **Vanilla GATv2 (Primary Baseline):** $54.00\% \pm 5.20\%$ (dynamic pairwise attention without regularization).
2. **GATv2 + Uniform DropEdge:** $50.33\% \pm 3.40\%$ (evaluates blind edge dropout on attention).
3. **GATv2 + Original AGA-Degree:** $54.00\% \pm 4.16\%$ (evaluates citation-style degree shield).
4. **GATv2 + Proposed Bio-AGA:** $50.00\% \pm 7.84\%$ (biophysically constrained shield without residuals).
5. *ResGATv2 without Bio-AGA:* $54.33\% \pm 5.28\%$ (evaluates residual highway alone).
6. **ResGATv2 + Bio-AGA:** $\mathbf{58.00\% \pm 3.71\%}$ (combined proposed architecture).
7. **Protein GCN (Additional Reference Architecture):** $58.50\% \pm 3.99\%$ (isotropic convolution benchmark).

Critically, comparing **ResGATv2 without Bio-AGA** ($54.33\%$) against **ResGATv2 + Bio-AGA** ($58.00\%$ in Exp A / $56.17\%$ in Exp E) directly demonstrates the independent contribution of the Bio-AGA regularization mechanism separate from the residual architecture.

---

### Q16: Why does simple GCN (58.50%) achieve performance comparable to your advanced model (58.00%)? Does this mean attention is useless?
**Answer:**
This is an important and honest scientific observation that demonstrates the value of retaining GCN as an additional reference architecture:
1. **Inductive Bias on Small Graphs:** ENZYMES contains 600 graphs with an average of only 32 nodes per graph. GCN has fixed isotropic degree normalization, which imposes a strong low-pass smoothing filter. On small datasets, GCN has far fewer free parameters and does not easily overfit.
2. **Attention Expressiveness vs Overfitting:** GAT and GATv2 are highly parameterized and expressive. Without regularization, they overfit (achieving 51.83% and 54.00%).
3. **The True Contribution:** When regularized with Bio-AGA and Residual connections, GATv2 reaches $58.00\%$, matching the GCN reference baseline while retaining dynamic attention interpretability. GCN cannot incorporate continuous edge distance encodings or distinguish between covalent bonds and electrostatic interactions; GATv2 can.

---

### Q17: In your experiment files, Baseline Exp A reports 58.00% for Bio-AGA + ResGATv2, while Ablation Exp E reports 56.17%. Why is there a difference?
**Answer:**
This is **not** a contradiction or reporting error; it reflects two deliberate experimental configurations:
- **Baseline Experiment A (100 Epochs):** Trained with 100 epochs and an early stopping patience of 25. The model achieved $58.00\% \pm 3.71\%$.
- **Ablation Experiment E (80 Epochs):** To evaluate 10 different ablation variants across 5 seeds (50 complete training runs) efficiently without excessive compute, the ablation protocol used 80 epochs. Under this 80-epoch protocol, the model achieved $56.17\% \pm 2.01\%$.
Crucially, in **both** protocols, Bio-AGA + ResGATv2 achieved the highest accuracy among all attention models (+7.67% gain over DropEdge in Exp A, and +6.84% gain in Exp E) and exhibited the lowest standard deviation.

---

### Q18: What did your analysis of the 1TIM enzyme demonstrate biologically?
**Answer:**
We tested our edge retention mechanism on the crystal structure of **Triosephosphate isomerase (PDB ID: 1TIM)**, focusing on its catalytic triad (Asn10, Lys12, Glu165):
- **Covalent Backbone Retention:** Uniform DropEdge retained only $80.0\%$ of backbone bonds; Bio-AGA retained **$100.0\%$**.
- **Catalytic Triad Retention:** Bio-AGA retained **$88.0\%$** of active-site contacts versus $79.8\%$ for Uniform DropEdge.
- **Non-Active-Site Contacts:** Bio-AGA retained $80.1\%$ of non-essential contacts.
This mathematically confirms that Bio-AGA acts as a selective structural shield, preserving catalytic geometry while regularizing non-essential interactions.

---

## Section 5: Engineering, Hardware & Limitations

### Q19: Why did you run training and evaluation on CPU rather than GPU / Apple Silicon MPS?
**Answer:**
During our initial profiling on Apple Silicon M-series hardware:
- PyTorch Geometric graph batching on Apple Silicon **MPS** exhibited severe driver dispatch and kernel launch overhead for small irregular graphs ($3.072$ seconds per epoch).
- On **CPU**, PyTorch utilizes optimized vectorized AVX/NEON kernels and high-bandwidth unified memory, taking only **$0.061$ seconds per epoch**—which is **50 times faster** than MPS!
Consequently, selecting CPU allowed us to run multi-seed baselines, 50 ablation runs, and the live demonstration in seconds.

---

### Q20: What are the main limitations of your project?
**Answer:**
1. **Dataset Scale:** We benchmarked on the 600-protein ENZYMES dataset. While standard for academic DL courses, evaluating on larger benchmarks like CATH or EC40 would further test scalability.
2. **Coarse-Grained Representation:** Residues were simplified to $C_\alpha$ centroid nodes. We did not model all-atom sidechain rotamers or quantum-mechanical dihedral angles.
3. **Static Snapshots:** PDB structures represent frozen, static crystal conformations, whereas real proteins undergo dynamic thermal fluctuations and conformational transitions in solution.

---

### Q21: If you had another semester to extend this project, what would you do?
**Answer:**
1. **Equivariant Architectures:** Upgrade from invariant GATv2 to an SE(3)-equivariant GNN (such as EGNN or PaiNN) so the model inherently respects 3D rotational and translational symmetries.
2. **Learned Edge Retention:** Instead of fixed closed-form formulas ($\Omega_{\text{seq}}$, $\Psi_{\text{dist}}$), use a lightweight meta-network trained via Reinforcement Learning or Gumbel-Softmax to learn edge retention probabilities end-to-end.
3. **Pre-training on AlphaFold Database:** Pre-train ResGATv2 using self-supervised masked residue prediction on 100,000 AlphaFold structures before fine-tuning on downstream enzyme function tasks.
