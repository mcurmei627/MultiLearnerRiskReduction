import numpy as np

def quadratic_min(subpops, i):
    b_list = []
    A_list = []
    for subpop in subpops:
        A, b = subpop.min_expr(i)
        A_list.append(A); b_list.append(b)
    A = np.sum(A_list, axis=0)
    b = np.sum(b_list, axis=0)
    return np.dot(np.linalg.pinv(A), b).flatten()

def noisy_quadratic_min(subpops, i):
    noiseless_theta = quadratic_min(subpops, i)
    var = np.var(noiseless_theta)
    noise = np.random.normal(size=noiseless_theta.shape, scale=0.0001*var)
    return noise+noiseless_theta

def learner_decisions(subpops, current = None, min_fn=quadratic_min):
    '''alpha[i,j] <- fraction of subpop j allocated to learner i'''
    n_learners = len(subpops[0].alphas)
    new_thetas = []
    for i in range(n_learners):
        if current is None or current == i:
            theta_i = min_fn(subpops, i)
            new_thetas.append(theta_i)
    return new_thetas

def get_all_risks(thetas, subpops, weighted_alpha=True, weighted_beta=True):
    n_learners = len(thetas)
    n_subpops = len(subpops)
    r = np.zeros((n_learners, n_subpops))
    for i in range(n_learners):
        for j in range(n_subpops):
            r[i,j] = subpops[j].risk(thetas[i])
            if weighted_alpha:
                r[i,j] *= subpops[j].alphas[i]
            if weighted_beta:
                r[i,j] *= subpops[j].beta
    return r

def subpop_decisions(thetas, subpops, epsilon = 0.1):
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
