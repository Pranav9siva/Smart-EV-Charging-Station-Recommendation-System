# Mathematical Formulation of Multi-Objective Reward & Lagrangian Costs

## 1. Multi-Objective Reward Function

For each EV agent $i$ selecting candidate station $a_i \in \{0, \dots, K-1\}$ at step $t$:

$$R_i(s_t, a_i) = R_{\text{success}} + R_{\text{arrival}} + R_{\text{recov}} - w_1 \cdot C_{\text{charge}} - w_2 \cdot T_{\text{wait}} - w_3 \cdot D_{\text{detour}} - w_4 \cdot E_{\text{cons}} - w_5 \cdot \text{Cost}_{\text{viol}}$$

### Reward Components:
1. **Successful Recommendation ($R_{\text{success}} = +10.0$)**: Awarded when a valid feasible charging station is recommended.
2. **Successful Arrival ($R_{\text{arrival}} = +15.0$)**: Awarded when the EV successfully reaches the station without depleting SOC below 5%.
3. **SOC Recovery ($R_{\text{recov}} = 0.5 \times (\text{SOC}_{\text{final}} - \text{SOC}_{\text{initial}})$)**: Proportional to net battery energy recovered.
4. **Charging Cost Penalty ($C_{\text{charge}} = \text{Energy}_{\text{need}} \times \text{Price}_{\text{per\_kWh}}$)**: Weighted by $w_1 = 0.05$.
5. **Waiting Time Penalty ($T_{\text{wait}} = \text{Queue}_{\text{len}} \times t_{\text{service}}$)**: Weighted by $w_2 = 0.2$.
6. **Detour Distance Penalty ($D_{\text{detour}} = d(\text{EV}, \text{Station}) - d(\text{EV}, \text{Dest})$)**: Weighted by $w_3 = 0.5$.
7. **Constraint Violation Cost ($\text{Cost}_{\text{viol}} = \sum_k \lambda_k c_k$)**: Dynamically weighted by learnable Lagrange multipliers $\lambda_k$.

---

## 2. Constraint Cost Signals ($c_{k, i}$)

1. **Safe SOC Reachability ($c_{1, i}$)**: $c_{1, i} = \max(0, 0.05 - \text{SOC}_{\text{arrival}})$
2. **Port Capacity Limit ($c_{2, i}$)**: $c_{2, i} = 1.0$ if $\text{FreePorts} = 0$, else $0.0$
3. **Queue Limit Threshold ($c_{3, i}$)**: $c_{3, i} = \max(0, \text{QueueLen} - \text{MaxQueueThreshold})$
4. **Detour Distance Limit ($c_{4, i}$)**: $c_{4, i} = \max(0, D_{\text{detour}} - D_{\text{max\_allowed}})$
5. **Grid Overload Threshold ($c_{5, i}$)**: $c_{5, i} = \max(0, \text{GridLoad}_{\text{kW}} - \text{GridCapacity}_{\text{kW}})$

---

## 3. Primal-Dual Lagrangian Update

$$\max_{\theta} \min_{\lambda \ge 0} \mathcal{L}(\theta, \lambda) = \mathbb{E}_{\theta} [R(\theta)] - \sum_{k=1}^{5} \lambda_k \left( \mathbb{E}_{\theta} [c_k(\theta)] - d_k \right)$$

Multiplier update step:
$$\lambda_k^{(t+1)} = \max\left(0, \lambda_k^{(t)} + \alpha_\lambda \cdot \left( \bar{c}_k - d_k \right)\right)$$
where $d_k$ is the constraint tolerance bound and $\alpha_\lambda$ is the dual learning rate.
