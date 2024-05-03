import numpy as np
from tqdm import tqdm


def quadratic_min(subpops, i, **kwargs):
    b_list = []
    A_list = []
    for subpop in subpops:
        A_mat, b = subpop.min_expr(i)
        A_list.append(A_mat)
        b_list.append(b)
    A_mat = np.sum(A_list, axis=0)
    b = np.sum(b_list, axis=0)
    return np.dot(np.linalg.pinv(A_mat), b).flatten()


def noisy_quadratic_min(subpops, i, **kwargs):
    noiseless_theta = quadratic_min(subpops, i)
    var = 1+np.var(noiseless_theta)
    noise = np.random.normal(size=noiseless_theta.shape, scale=0.01*var)
    return noise+noiseless_theta


def gradient_step(subpops, i, theta=0, step_size=0.01):
    '''gradient descent on the risk function'''
    b_list = []
    A_list = []
    for subpop in subpops:
        A, b = subpop.min_expr(i)
        A_list.append(A)
        b_list.append(b)
    A = np.sum(A_list, axis=0)
    b = np.sum(b_list, axis=0)
    if theta is None:
        theta = 0
    theta = theta - step_size*(A@theta - b)
    return theta


def learner_decisions(subpops, current=None, current_thetas=None, min_fn=quadratic_min):
    '''alpha[i,j] <- fraction of subpop j allocated to learner i'''
    n_learners = len(subpops[0].alphas)
    new_thetas = []
    for i in range(n_learners):
        if current is None or current == i:
            theta_i = min_fn(subpops, i, theta=current_thetas[i])
            new_thetas.append(theta_i)
    return new_thetas


def get_all_risks(thetas, subpops, weighted_alpha=True, weighted_beta=True):
    n_learners = len(thetas)
    n_subpops = len(subpops)
    r = np.zeros((n_learners, n_subpops))
    for i in range(n_learners):
        for j in range(n_subpops):
            if subpops[j].kind == 'sampled_shared':
                r[i, j] = subpops[j].risk(thetas[i], population=True)
            else:
                r[i, j] = subpops[j].risk(thetas[i])
            if weighted_alpha:
                r[i, j] *= subpops[j].alphas[i]
            if weighted_beta:
                r[i, j] *= subpops[j].beta
    return r


def subpop_decisions(thetas, subpops, epsilon=0.1):
    [subpop.update_alpha(thetas, epsilon) for subpop in subpops]
    return np.array([subpop.alphas for subpop in subpops])


def average_risk_subpop(thetas, subpops):
    r = get_all_risks(thetas, subpops, weighted_alpha=True)
    return np.sum(r, 0)


def average_risk_learner(thetas, subpops):
    r = get_all_risks(thetas, subpops, weighted_alpha=True, weighted_beta=True)
    return np.sum(r, 1)


def learner_share(subpops):
    n_learners = len(subpops[0].alphas)
    return [np.sum([s.alphas[i]*s.beta for s in subpops]) for i in range(n_learners)]


def convergent_count(subpops):
    c_counts = [subpop.converged_for for subpop in subpops]
    return c_counts


def run_experiment(T, subpops, min_fn=quadratic_min, early_stop=False, verbose=True, sample_size=None, temp=0.1):
    """Simulate the experiment for T time steps. Return the average risks for each subpopulation and learner, 
        as well as the risks for each learner and subpopulation at each time step."""
    n_learners = len(subpops[0].alphas)
    if verbose:
        print("Initial Conditions:")
        print(f"There are {n_learners} initial learners")
        print(f"Subpopulation splits: {[s.beta for s in subpops]}")
        print(f"Initial allocations: {[s.alphas for s in subpops]}")
        print(
            f"Optimal decisions theta for each subpop (row-wise) {[s.phi for s in subpops]}")

    average_risks_subpop = []
    average_risks_learner = []
    all_risks = []
    all_thetas = []
    all_alphas = []
    thetas = np.zeros((n_learners, len(subpops[0].phi)))

    for t in tqdm(range(T)):
        for subpop in subpops:
            if (subpop.kind == 'sampled_shared' or subpop.kind == 'sampled') and sample_size is not None:
                subpop.sample(sample_size)

        thetas = np.array(learner_decisions(
            subpops, current_thetas=thetas, min_fn=min_fn))
        alpha = subpop_decisions(thetas, subpops, epsilon=temp)
        all_thetas.append(thetas)
        all_alphas.append(alpha.T)
        risks = get_all_risks(thetas, subpops)
        all_risks.append(risks)
        a_risk_subpop = average_risk_subpop(thetas, subpops)
        average_risks_subpop.append(a_risk_subpop)
        a_risk_learner = average_risk_learner(thetas, subpops)
        average_risks_learner.append(a_risk_learner)

        if early_stop:
            c_counts = convergent_count(subpops)
            if min(c_counts) > 20:
                break

    if verbose:
        print("Final Conditions:")
        print(f'\t\tLearners decisions: {thetas}')
        print(f'\t\tSubpopulation allocations: {alpha}')

    return average_risks_subpop, average_risks_learner, all_risks, all_thetas, all_alphas


def run_competition_experiment(T, subpops, n_learners_init=2, n_learners_max=None, min_fn=noisy_quadratic_min):
    n_learners_init = len(subpops[0].alphas)
    if n_learners_max is None:
        n_learners_max = len(subpops)
    exp_out = run_experiment(T, subpops, early_stop=True,
                             verbose=False, min_fn=min_fn)
    average_risks_subpop = exp_out[0]
    total_average_risks_subpop_over_time = {-1: average_risks_subpop}
    for i in range(n_learners_max - n_learners_init):
        for subpop in subpops:
            subpop.break_learner(i)
        exp_out = run_experiment(
            T, subpops, early_stop=True, verbose=False, min_fn=min_fn)
        total_average_risks_subpop_over_time[i] = exp_out[0]
    return (total_average_risks_subpop_over_time)
