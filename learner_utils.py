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
        if current is None or current == i or i in current:
            
            theta_i = min_fn(subpops, i)
            new_thetas.append(theta_i)
    return new_thetas

def learner_max_price(thetas, prices, subpops, min_fn=quadratic_min):
    n_learners = len(thetas)
    new_prices = [0] * len(prices)
    #price_candidates = np.linspace(0, 100, 51)
    for i in range(n_learners):
        price_candidates = [prices[i] + 1, prices[i] * 2, prices[i] / 2]
        profit_candidates = []
        for p in price_candidates:
            tmp = prices[i]
            prices[i] = p
            new_alphas = subpop_decisions(thetas, subpops, epsilon=0.1, prices=prices, dryrun=True)
            profit_candidates.append(new_alphas[:, i].sum() * p)  # warning: assumes beta is uniform
            prices[i] = tmp
        new_prices[i] = price_candidates[np.argmax(profit_candidates)]
    return new_prices

def learner_price_decisions(thetas, prices, subpops, min_fn=quadratic_min):
    '''alpha[i,j] <- fraction of subpop j allocated to learner i'''
    n_learners = len(thetas)
    new_prices = []
    new_thetas = []
    learner_risks = average_risk_learner(thetas, subpops)
    for i in range(n_learners):
        theta_i = min_fn(subpops, i)
        new_thetas.append(theta_i)
    new_learner_risks = average_risk_learner(new_thetas, subpops)
    for i in range(n_learners):
        surplus = learner_risks[i] - new_learner_risks[i]
        new_prices.append(prices[i] + surplus)
    # print('old thetas', thetas, 'old learner risks', learner_risks)
    # print('new thetas', new_thetas, 'new learner risks', new_learner_risks)
    return new_thetas, new_prices

def get_all_risks(thetas, subpops, weighted_alpha=True, weighted_beta=True, prices=None):
    n_learners = len(thetas)
    if prices is None:
        prices = np.zeros(n_learners)
    n_subpops = len(subpops)
    r = np.zeros((n_learners, n_subpops))
    for i in range(n_learners):
        for j in range(n_subpops):
            r[i,j] = subpops[j].risk(thetas[i], prices[i])
            if weighted_alpha:
                r[i,j] *= subpops[j].alphas[i]
            if weighted_beta:
                r[i,j] *= subpops[j].beta
    return r

def subpop_decisions(thetas, subpops, epsilon = 0.1,  prices=None, dryrun=False):
    n_learners = len(thetas)
    if prices is None:
        prices = np.zeros(n_learners)
    alphas = [subpop.update_alpha(thetas, epsilon, prices, dryrun=dryrun) for _, subpop in enumerate(subpops)]
    return np.array(alphas)

def average_risk_subpop(thetas, subpops, prices=None):
    r = get_all_risks(thetas, subpops, weighted_alpha=True, prices=prices)
    return np.sum(r, 0)

def average_risk_learner(thetas, subpops, prices=None):
    r = get_all_risks(thetas, subpops, weighted_alpha=True, weighted_beta=True, prices=prices)
    return np.sum(r, 1)

def learner_share(subpops):
    n_learners = len(subpops[0].alphas)
    return [np.sum([s.alphas[i]*s.beta for s in subpops]) for i in range(n_learners)]

def convergent_count(subpops):
    c_counts = [subpop.converged_for for subpop in subpops]
    return c_counts
