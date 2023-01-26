# Multi Player Risk Reduction

This repository contains the numerical experiments for **Emergent segmentation from participation dynamics and multi-learner retraining** paper currently under review @ICML 2023.

## Setup

We assume there are $M$ competing firms, which we will call _learners_. Similarly we have $N$ subpopulations such that $\mathcal{D}_i$ is the distribution of users in subpopulation $i$. Let $\beta_i$ be the relative size of the population.

We instantiate the following sequential updates:

- At time $t$: $\alpha^t_j \in \Delta^{N}$ is the allocation vector such that $\alpha_{ij}^t$ is the fraction of subpopulation $i$ allocated to learner $j$ at time $t$.

- Learner $j$ observes a distribution of users: $\mathbb{D}_j^t \propto \sum_{i=1}^N \beta_i \alpha_{ij}^t \mathcal{D}_i$.

- Learner $j$ selects a decision rule parametrized by $\theta_j^t$. One way to do that is to assume that the learner perform risk minimization. At the population level (as opposed to finite samples) this would be:

  $$\theta^t = \arg\min_{\theta} \mathbb{E}_{x\sim \mathbb{D}\_j^t}[\ell(x, \theta)] =\arg \min_{\theta}\sum_{i=1}^N \beta_i\alpha^t_{ij} \mathcal{R}_i(\theta)$$

  where $\mathcal{R}_i(\theta) = \mathbb{E}_{x\sim \mathcal{D}\_i}[\ell(x, \theta)]$ and $\ell$ is some loss function.

If we make the assumption that the average risk experienced by subpopulation $i$ is quadratic $\mathcal{R}_i(\theta) = \theta^T A_i \theta + b_i\theta + c_i$. Further assume that $A_i$ is positive definite and define $\phi_i:=(A_i^T A_i)^{-1} A_i b_i$ is the optimal risk minimizing decision for subpopulation $i$.

- Given decisions of the learners, each subpopulation experiences average risk: $R^{\texttt{subpop}}_i = \sum_{j=1}^M \alpha_{ij}^t \mathcal{R}_i(\theta^t_j)$. Based on the risk with respect to each learner, the subpopulation updates the allocations to $\alpha_i^{t+1}$.
- We study Multiplicative Weightw Updade Dynamics (MWUD) of the form:

$$\alpha_{ij}^{t+1} \propto \alpha_{ij}^t(1-\epsilon)^{c_i(r_i(\theta_j^t))} $$
where $c_j$ is some sensible choice of comparison function, in experiments we will use the identity.

## Experiments

We study the resulting decision dynamics. The first experimental result illustrates the non-monotonicity of average risks for learners and subpopulations. Further we illustrate that despite individual non-monotonicity the total loss is non-increases, and thus serves as a potential function for the decision dynamics. Second set of experiments investigate the social welfare benefits of increased competition among learners.

All plots in the paper can be obtained by running the accompanying notebook. Replicating the experiments should take under 5 minutes in terms of running time on a standard laptop.
