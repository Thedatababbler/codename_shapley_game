# PNS-Based Baseline + Normalized Surplus Reward Redistribution Spec

## 1. Objective

Implement a **budget-preserving step-level reward redistribution module** for reasoning RL / GRPO.

The module takes:

- a trajectory-level final reward `R`
- step-level PNS outputs for each reasoning step

and redistributes `R` into per-step rewards such that:

1. **Reward conservation** holds:
   \[
   \sum_{t=1}^{T} r_t = R
   \]

2. Steps with higher PNS receive more reward.

3. Redistribution is **not hand-tuned via arbitrary per-step multipliers**, but derived from a normalized credit-sharing rule.

The default method is:

- assign every step a uniform **baseline share**
- redistribute the remaining reward budget according to **normalized surplus** over step-level PNS scores

---

## 2. Core Formulation

Given a trajectory:

\[
\tau = (s_1, s_2, \dots, s_T)
\]

with final reward:

\[
R
\]

and step-level PNS scores:

\[
c_1, c_2, \dots, c_T
\]

we define the final step reward as:

\[
r_t = (1-\alpha)\frac{R}{T} + \alpha R \phi_t
\]

where:

- \(T\) = number of steps in the trajectory
- \(\alpha \in [0,1]\) = interpolation coefficient between uniform allocation and PNS-aware redistribution
- \(\phi_t\) = normalized surplus share for step \(t\)

This guarantees:

\[
\sum_{t=1}^{T} r_t = R
\]

---

## 3. Step Score Construction

### 3.1 If the PNS model is regression-based

Use the raw scalar score directly:

\[
c_t = \text{raw\_score}_t
\]

### 3.2 If the PNS model is classification-based

Suppose the classifier outputs a probability vector over \(K\) discrete PNS classes:

\[
p_t(0), p_t(1), \dots, p_t(K-1)
\]

Map each class to a numeric value:

\[
v_0, v_1, \dots, v_{K-1}
\]

For example, if `k=3` alternatives and classes correspond to \(\{0, 1/3, 2/3, 1\}\), set:

\[
v = [0, \tfrac13, \tfrac23, 1]
\]

Then compute the expected PNS score:

\[
c_t = \sum_{k=0}^{K-1} p_t(k)\, v_k
\]

This expected value is the step-level scalar score used by the redistribution module.

---

## 4. Normalized Surplus Share

### 4.1 Surplus definition

For each trajectory, compute:

\[
c_{\min} = \min_{j=1,\dots,T} c_j
\]

Then define the surplus of step \(t\) as:

\[
u_t = \max(c_t - c_{\min}, 0)
\]

Interpretation:

- the lowest-scoring step(s) define the local baseline
- only the amount above the minimum competes for extra reward

### 4.2 Normalization

If:

\[
\sum_t u_t > 0
\]

then define:

\[
\phi_t = \frac{u_t}{\sum_j u_j}
\]

Else, if all step scores are identical:

\[
\phi_t = \frac{1}{T}
\]

for all \(t\).

Thus:

- \(\phi_t \ge 0\)
- \(\sum_t \phi_t = 1\)

---

## 5. Final Allocation Rule

### Main version

\[
r_t = (1-\alpha)\frac{R}{T} + \alpha R \phi_t
\]

Interpretation:

- the first term is a uniform baseline
- the second term redistributes reward according to normalized surplus

### Properties

1. **Conservation**
   \[
   \sum_t r_t = R
   \]

2. **Monotonicity**
   Larger PNS scores generally imply larger redistributed shares.

3. **Stability**
   When PNS scores are flat, allocation falls back to uniform.

---

## 6. Integration Options

### Option A: Step-level rewards

Use `r_t` as the per-step reward.

This is appropriate if the RL training pipeline supports step-level rewards directly.

### Option B: Step-level advantages

If the framework already computes a trajectory-level or group-level advantage \(A\), define:

\[
A_t = (1-\alpha)\frac{A}{T} + \alpha A \phi_t
\]

This is the exact analogue of the reward rule, but applied to advantages instead of rewards.

---

## 7. Token-Level Mapping

If optimization happens at token level, broadcast each step allocation to the tokens belonging to that step.

### Default implementation

- assume each reasoning step has a token span
- assign the same `r_t` (or `A_t`) to all tokens inside that step

### Optional refinement

If desired, divide the step reward uniformly across tokens in the step.  
For the first implementation, simple broadcasting is preferred.

---

## 8. Required Inputs

For each trajectory, the implementation must support:

- `steps`: list of step strings
- `final_reward`: scalar reward `R`
- `pns_output`:
  - regression mode: list/array of shape `[T]`
  - classification mode: list/array of shape `[T, K]`
- optional `step_token_spans`: mapping from step index to token span

### Config parameters

- `alpha` (default: `0.5`)
- `eps` (default: `1e-8`)
- `mode` in `{"regression", "classification"}`
- `pns_values` for classification mode  
  default example:
  ```python
  [0.0, 1/3, 2/3, 1.0]
  ```

---

## 9. Implementation Requirements

Implement the following functions.

### 9.1 `expected_pns_score`

**Purpose:** convert model outputs into scalar step scores.

#### Signature
```python
def expected_pns_score(step_logits_or_probs, pns_values=None, mode="classification"):
    ...
```

#### Behavior

- if `mode == "regression"`: return the raw scores
- if `mode == "classification"`:
  - assume probabilities or convert logits to probabilities
  - return the expected scalar value per step

---

### 9.2 `compute_normalized_surplus_shares`

**Purpose:** compute \(\phi_t\) from scalar step scores.

#### Signature
```python
def compute_normalized_surplus_shares(scores, eps=1e-8):
    ...
```

#### Behavior

1. compute `c_min`
2. compute `surplus[t] = max(scores[t] - c_min, 0)`
3. if total surplus is zero, return uniform shares
4. otherwise return normalized surplus shares

---

### 9.3 `allocate_reward`

**Purpose:** produce final per-step allocation from shares.

#### Signature
```python
def allocate_reward(final_reward, shares, alpha=0.5):
    ...
```

#### Behavior

For each step:

\[
r_t = (1-\alpha)\frac{R}{T} + \alpha R \phi_t
\]

---

### 9.4 End-to-end wrapper

#### Signature
```python
def pns_baseline_normalized_surplus_allocation(
    final_reward,
    pns_output,
    mode="classification",
    pns_values=None,
    alpha=0.5,
    eps=1e-8,
):
    ...
```

#### Return fields

Return a dict containing at least:

- `"scores"`: scalar PNS score per step
- `"shares"`: normalized surplus shares
- `"step_rewards"`: final per-step allocated rewards

---

## 10. Reference Pseudocode

```python
import torch

def expected_pns_score(step_logits_or_probs, pns_values=None, mode="classification"):
    if mode == "regression":
        return step_logits_or_probs

    probs = step_logits_or_probs  # assume probabilities
    scores = []
    for t in range(len(probs)):
        s = 0.0
        for k in range(len(pns_values)):
            s += probs[t][k] * pns_values[k]
        scores.append(float(s))
    return scores


def compute_normalized_surplus_shares(scores, eps=1e-8):
    T = len(scores)
    c_min = min(scores)
    surplus = [max(s - c_min, 0.0) for s in scores]
    total_surplus = sum(surplus)

    if total_surplus <= eps:
        return [1.0 / T for _ in range(T)]

    return [u / total_surplus for u in surplus]


def allocate_reward(final_reward, shares, alpha=0.5):
    T = len(shares)
    uniform = 1.0 / T
    rewards = []
    for phi in shares:
        r = (1 - alpha) * final_reward * uniform + alpha * final_reward * phi
        rewards.append(r)
    return rewards


def pns_baseline_normalized_surplus_allocation(
    final_reward,
    pns_output,
    mode="classification",
    pns_values=None,
    alpha=0.5,
    eps=1e-8,
):
    scores = expected_pns_score(
        pns_output,
        pns_values=pns_values,
        mode=mode,
    )
    shares = compute_normalized_surplus_shares(scores, eps=eps)
    step_rewards = allocate_reward(final_reward, shares, alpha=alpha)

    return {
        "scores": scores,
        "shares": shares,
        "step_rewards": step_rewards,
    }
```

---

## 11. Recommended Defaults

Use the following defaults for the first implementation:

```python
alpha = 0.5
eps = 1e-8
pns_values = [0.0, 1/3, 2/3, 1.0]
```

### Notes

- `alpha = 0.5` provides a balanced interpolation
- increase `alpha` only after confirming training stability
- keep the first version simple; avoid temperature transforms or extra normalization tricks initially

---

## 12. Sanity Checks

The implementation must pass the following checks.

### Check 1: Reward conservation

For every trajectory:

\[
\sum_t r_t \approx R
\]

### Check 2: Uniform fallback

If all step scores are identical, then:

\[
r_t = \frac{R}{T}
\]

for all \(t\).

### Check 3: Monotonicity trend

If one step has a substantially larger score than others, its final allocated reward should be larger.

### Check 4: Alpha endpoints

- when `alpha = 0`, the result must reduce to uniform allocation
- when `alpha = 1`, the result must be fully determined by normalized surplus shares

---

## 13. Logging Requirements

For debugging and analysis, log the following per batch or averaged over an epoch:

- mean/std of scalar step scores
- mean/std of surplus values
- entropy of allocation shares
- maximum share per trajectory
- percentage of trajectories falling back to uniform allocation
- correlation between raw PNS scores and final allocated rewards

These logs are important for diagnosing overly sharp or overly flat redistribution.

---

## 14. Ablation Variants to Implement

Please implement the following three variants for ablation.

### Variant A: Uniform baseline
\[
r_t = \frac{R}{T}
\]

### Variant B: Direct normalized score
\[
\phi_t = \frac{c_t}{\sum_j c_j}
\]
\[
r_t = (1-\alpha)\frac{R}{T} + \alpha R \phi_t
\]

### Variant C: Baseline + normalized surplus (main method)
\[
u_t = \max(c_t - c_{\min}, 0)
\]
\[
\phi_t =
\begin{cases}
u_t / \sum_j u_j, & \sum_j u_j > 0 \\
1/T, & \text{otherwise}
\end{cases}
\]
\[
r_t = (1-\alpha)\frac{R}{T} + \alpha R \phi_t
\]

Variant C is the default and main method.

---

## 15. Deliverables

The coding agent should produce:

1. a standalone Python module implementing the redistribution methods
2. unit tests for the sanity checks
3. a simple demo script showing:
   - input PNS outputs
   - computed scalar scores
   - surplus shares
   - final per-step rewards
4. hooks or utilities to integrate the allocation into the current RL / GRPO training pipeline

---

## 16. Recommended Naming

Use one of the following names in code/comments:

- `pns_reward_redistributor`
- `normalized_surplus_allocator`
- `pns_credit_sharing`
- `pns_reward_allocation`

Preferred method name:
```python
pns_baseline_normalized_surplus_allocation
```

---

## 17. Summary

This module should implement a **PNS-based reward sharing rule** that:

- turns step-level PNS signals into scalar scores
- converts scores into normalized surplus shares
- redistributes a trajectory-level reward under a fixed reward budget
- avoids arbitrary hand-tuned multipliers
- provides a simple and stable first implementation for PNS-aware credit assignment
