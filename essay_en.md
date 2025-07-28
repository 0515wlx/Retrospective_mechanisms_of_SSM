Theories may not be correct

### Theoretical Possibility of Inaccuracy: Impact of Retrospective Mechanisms on Polynomial-Time Capabilities in State Space Models

#### Abstract
This paper establishes a rigorous theoretical framework for the computational complexity of State Space Models (SSMs). We demonstrate that:  
1. For fixed state dimension \(d\), basic SSMs (without retrospective mechanisms) can only solve problems with constant space complexity (equivalent to finite automata).  
2. With retrospective mechanisms, SSMs can solve any Linear Bounded Automaton (LBA) problem (equivalent to PSPACE).  

**Key conclusions**:  
1. **Capability Boundaries**:  
   - SSMs with retrospective mechanisms = PSPACE  
   - SSMs without retrospective mechanisms = REG  
2. **Time Inflation**: When metadata demand \(\text{MD}(P,n) > \alpha d\), the time lower bound is \(\Omega(n \cdot \min(2^{\Delta m}, \Delta m))\) (\(\Delta m = \text{MD}(P,n) - \alpha d\)).  
3. **Equivalence Condition**: If \(\text{MD}(P,n) \leq \alpha d\), both SSMs with/without retrospective mechanisms can solve the problem in \(O(n)\) time.  
4. **Architectural Implications**: State dimension \(d\) determines the amount of storable metadata; retrospective mechanisms compensate for spatial limitations through time overhead.  

---

### 1 Introduction
State Space Models (SSMs) represent a new paradigm for sequence modeling, yet their theoretical foundations remain underspecified. This paper addresses the following gaps:  
- **Rigorous Formalization**: Establish equivalence between SSMs and Linear Bounded Automata (LBAs).  
- **Complexity Lower Bounds**: Provide tight lower bounds for time inflation (\(\Omega(n \cdot \min(2^{\Delta m}, \Delta m))\)).  
- **Capability Stratification**: Reveal the decisive role of state dimension \(d\) in computational capabilities.  
- **Experimental Validation**: Verify theoretical predictions through modular \(n\) computation tasks.  

---

### 2 Model Definitions

#### 2.1 Basic SSM Formalization
- **State Update**:  
  \[
  s_t = A s_{t-1} + B x_t \quad (s_t \in \mathbb{R}^d, A,B \text{ learnable})
  \]
- **Output**:  
  \[
  y_t = C s_t + D x_t
  \]

#### 2.2 Retrospective Mechanism Extensions
- **Pointer Generation**:  
  \[
  i_t = \text{ptr}(s_t, t) \in \{1,2,\dots,n\}
  \]
- **Content Access**:  
  \[
  y_t = g(s_t, x_{i_t}) \quad \text{or} \quad y_t = g(s_t, X)
  \]
- **Overhead Model**:  
  - Index access: \(\Theta(1)\) time  
  - Content matching: \(\Theta(n)\) time (requires sequence scan)  

#### 2.3 Problem Complexity Metrics
- **Metadata Demand** \(\text{MD}(P,n)\): Minimum number of control state bits required to solve problem \(P\).  
  *Examples*:  
  - Parity check: \(\text{MD}=1\)  
  - Integer multiplication: \(\text{MD}=\Omega(\log n)\)  

---

### 3 Capability Boundaries of SSMs Without Retrospection

#### Theorem 3.1 (Computational Equivalence)
The computational capability of SSMs without retrospective mechanisms is equivalent to Deterministic Finite Automata (DFAs), i.e., they can only recognize regular languages (REG class).  

**Proof**:  
Let the SSM state space be \(\mathcal{S} \subseteq \mathbb{R}^d\), discretized to \(|\mathcal{S}| \leq 2^{\alpha d}\). Construct a DFA:  
- State set \(Q = \mathcal{S}\)  
- Transition function \(\delta(s, x) = \lfloor A s + B x \rfloor_{\text{discrete}}\)  
- Accepting states \(F = \{ s \mid C s \geq 0 \}\)  
Then \(L(\text{SSM}) = L(M)\). The reverse equivalence follows by encoding DFA states. ∎  

#### Corollary 3.1
SSMs without retrospective mechanisms can solve problems if and only if \(\text{MD}(P,n) \leq \alpha d\), with time complexity \(\Theta(n)\).  

---

### 4 Capabilities and Costs of SSMs With Retrospection

#### Theorem 4.1 (Computational Equivalence)
The computational capability of SSMs with retrospective mechanisms is strictly equivalent to Linear Bounded Automata (LBAs), with capability range PSPACE.  

**Proof**:  
1. **(⊆ LBA)** The SSM state space size \(|\mathcal{S}| = 2^{\alpha d}\), combined with input sequence \(X \in \Sigma^n\), yields total states \(2^{\alpha d} \cdot |\Sigma|^n\), equivalent to \(O(n)\)-space Turing machines.  
2. **(⊇ LBA)** For any LBA (with \(O(n)\)-space restriction), construct an SSM:  
   - State \(s_t\) stores working tape content (requires \(d = \Theta(n)\))  
   - Retrospective mechanism implements read/write head operations  
   - Per-step time overhead \(O(n)\), total time \(O(T(n) \cdot n)\) ∎  

#### Theorem 4.2 (Tight Time Inflation Lower Bound)
If \(\text{MD}(P,n) > \alpha d\), the time lower bound is:  
\[
T(n) = \Omega\left( n \cdot \min\left(2^{\text{MD}(P,n) - \alpha d},  \frac{\text{MD}(P,n)}{\alpha d} \right) \right)
\]

**Proof**:  
1. **Exponential Lower Bound**: When \(\Delta m = \Omega(\log n)\), \(2^{\Delta m}\) scans are required (\(\Omega(n \cdot 2^{\Delta m})\)).  
2. **Linear Lower Bound**: When \(\Delta m = o(\log n)\), \(\Omega(\Delta m)\) scans are required (\(\Omega(n \cdot \Delta m)\)). ∎  

#### Figure 1: Stratification of State Dimension and Computational Capabilities
```
Computational Capability
^
│ PSPACE    (d = Θ(n))
│   │
│ NP        (d = Θ(n^k), k≥1)
│   │  
│ P         (d = Θ(log n))
│   ├─────┤  
│ REG      (d = O(1))
└─────────→ 
```

---

### 5 Experimental Validation

#### 5.1 Experimental Setup
- **Task**: Modular \(n\) computation (input: binary sequence, output: sequence value \(\mod n\)).  
- **Models**:  
  - Basic SSM: No retrospective mechanism  
  - Retro-SSM: With retrospective mechanism  
- **Metrics**: Bit accuracy, numerical accuracy  
- **Parameters**: \(d_{\text{model}}=4, d_{\text{state}}=4\)  

#### 5.2 Results Analysis

##### Table 1: Accuracy Comparison Across Different \(n\) Values (%)
| \(n\)   | \(2^1\) | \(2^2\) | \(2^3\) | \(2^4\) | \(2^5\) | ... | \(2^{16}\) |
|---------|---------|---------|---------|---------|---------|-----|------------|
| No Retrospection | 100.00  | 100.00  | 88.73   | 84.47   | 78.23   | ... | 62.86      |
| With Retrospection | 93.08   | 96.49   | 96.88   | 94.94   | 93.56   | ... | 87.02      |

- Specific results: result.txt  

**Key Findings**:  
1. **Capability Boundary Validation**:  
   - When \(n \leq 4\) (\(\text{MD}=2 \leq \alpha d\)), non-retrospective SSMs achieve 100% accuracy.  
   - When \(n > 4\) (\(\text{MD} > \alpha d\)), non-retrospective SSM accuracy drops sharply.  

2. **Time Cost Manifestation**:  
   - Retrospective mechanisms enable SSMs to handle arbitrary \(n\), but training time increases significantly with \(n\).  
   - For \(n = 2^{16}\), Retro-SSM training time is 60x longer than for \(n=2^4\) (estimated, not precisely measured).  

3. **S4 Model Characteristics**:  
   - Increasing \(d\) to \(\Theta(\log n)\) improves accuracy by <3% for both SSM types.  

4. **Github Link**:
   - https://github.com/0515wlx/Retrospective_mechanisms_of_SSM

---

### 6 Conclusion

1. **Capability Essence**:  
   - Non-retrospective SSMs ≡ Finite Automata (REG class)  
   - Retrospective SSMs ≡ Linear Bounded Automata (PSPACE class)  

2. **Cost Constraints**:  
   - Tight time inflation lower bound \(\Omega(n \cdot \min(2^{\Delta m}, \Delta m))\)  
   - State dimension \(d\) determines storable metadata capacity  

3. **Design Implications**:  
   - For low \(\text{MD}\) problems, prioritize non-retrospective SSMs  
   - For high \(\text{MD}\) problems, require \(d = \Theta(\log n)\) or accept time inflation  

This work reveals the fundamental limitations of SSMs in long-sequence modeling—the trade-off between state dimension and computational capability—providing a theoretical foundation for future architectural innovations.  

---

**Appendix: Supplementary Proofs**

**Detailed Proof of Theorem 4.1**  
Let the problem \(P\) require distinguishing a configuration set \(\mathcal{C}\) with \(|\mathcal{C}| = N = 2^m\), \(m = \text{MD}(P,n)\). The SSM state space is \(|\mathcal{S}| = K = 2^{\alpha d}\).  

*Step 1*: Partition the input sequence into \(r\) segments \(X_1,\dots,X_r\), each of length \(l = \lfloor n/r \rfloor\).  
*Step 2*: Proof by contradiction. If total scans \(< r_0 = \lceil N/K \rceil\), then the number of state transition paths \(< K \cdot r_0 < N\).  
*Step 3*: Each scan requires at least \(l = \Omega(n/r_0)\) steps.  
*Step 4*: Total time lower bound:  
\[
T(n) \geq r_0 \cdot l = \Omega(n \cdot \min(2^{\Delta m}, \Delta m))
\]∎

**References**  
[1] Gu A. et al. Efficiently Modeling Long Sequences with Structured State Spaces. *arXiv:2111.00396*, 2022.