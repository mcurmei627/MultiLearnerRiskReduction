import numpy as np
import pandas as pd
import folktables
from sklearn.linear_model import LinearRegression


class SubPop():
    def __init__(self, phi, beta, alphas, cov=None, sigmasq=1):
        """ Subpopulation Class

        Args:
            phi (np.array): optimal decision rule for the sub-population
            beta (float): Size of the subpopulation (absolute or relative)
            alphas (np.array): Allocation of the sub-population to different firms
                                must sum up to 1
            cov (2d np.array, optional): Covariance matrix. Defaults to None.
            sigmasq (float. optional): Risk offset. Defaults to 1.
        """
        self.phi = phi
        self.d = phi.size
        self.cov = np.eye(self.d) if cov is None else cov
        self.sigmasq = sigmasq
        self.beta = beta
        self.alphas = alphas
        self.t = 0
        self.converged = False
        self.converged_for = 0

    def update_alpha(self, thetas, epsilon=0.1,):
        """Updates the allocations of the subpopulation

        Args:
            thetas (iterable of np.array): the decisions of each learner
            epsilon (float, optional): constant on the MWUD. Defaults to 0.1.
        """
        r = [self.risk(theta) for theta in thetas]
        new_alphas = self.alphas * np.power((1-epsilon), r)
        new_alphas /= np.sum(new_alphas)
        self.converged = np.allclose(new_alphas, self.alphas)
        if self.converged:
            self.converged_for += 1
        self.alphas = new_alphas
        self.t += 1

    def break_learner(self, i):
        """Breaks learner i into two different learners by adding a clone of the learner
        to the end of the list of learner

        Args:
            i (int): index of learner to be split
        """
        alphas = self.alphas
        learner_alphas = alphas[i]/2
        # slightly perturb the learner's allocation
        original_learner_alphas = learner_alphas + \
            np.random.normal(0, 0.001)
        # clip between 0 and 1
        original_learner_alphas = np.clip(original_learner_alphas, 0, 1)
        new_learner_alphas = alphas[i] - original_learner_alphas
        alphas[i] = original_learner_alphas
        self.alphas = np.append(alphas, new_learner_alphas)
        self.converged = False
        self.converged_for = 0

    def risk(self, theta):
        """Computes the average risk for the subpopulation with respect to parameter theta

        Args:
            theta (np.array): decision of a learner
        """
        pass


class QuadraticSubPop(SubPop):
    def __init__(self, *args, **kwargs):
        super(QuadraticSubPop, self).__init__(*args, **kwargs)
        self.kind = 'quadratic'

    def min_expr(self, i):
        """Helper function used by learner i in the minimization of the risk
        Args:
            i (int): index of learner

        Returns:
            A: scaled covariance metric
            b: scaled cov^T*phi
        """
        A = self.beta * self.alphas[i] * self.cov
        b = self.beta * self.alphas[i] * np.dot(self.cov, self.phi)
        return A, b

    def risk(self, theta):
        """Computes the average risk for the subpopulation assumong that is  quadradic

        Args:
            theta (np.array): decision of a learner
        """
        return (np.linalg.norm(np.dot(self.cov, theta - self.phi))**2 + self.sigmasq)


class EmpiricalSubPop(SubPop):
    def __init__(self, xs, ys, *args,  **kwargs):
        self.xs = xs
        self.ys = ys
        self.N = self.xs.shape[0]
        phi_emp = LinearRegression(
            fit_intercept=False).fit(self.xs, self.ys).coef_
        super(EmpiricalSubPop, self).__init__(phi_emp, *args, **kwargs)
        self.cov = self.xs.T @ self.xs/self.N
        self.kind = 'empirical'

    def min_expr(self, i):
        A = self.beta*self.alphas[i] * self.xs.T @  self.xs
        b = self.beta*self.alphas[i] * self.xs.T @ self.ys
        return A, b

    def risk(self, theta, relative=False):
        risk = np.linalg.norm(self.xs @ theta - self.ys)**2/self.N
        if not relative:
            return risk
        min_risk = np.linalg.norm(self.xs @ self.phi - self.ys)**2/self.N
        return risk/min_risk


class SampledSubPop(SubPop):
    def __init__(self, xs, ys, *args,  **kwargs):
        self.xs = xs
        self.ys = ys
        self.N = self.xs.shape[0]
        phi_emp = LinearRegression(
            fit_intercept=False).fit(self.xs, self.ys).coef_
        super(SampledSubPop, self).__init__(phi_emp, *args, **kwargs)
        self.kind = 'sampled_shared'
        self.pop_N = self.N
        self.pop_xs = self.xs
        self.pop_ys = self.ys

    def sample(self, n):
        idx = np.random.choice(self.pop_N, size=n, replace=False)
        self.N = n
        self.xs = self.pop_xs[idx]
        self.ys = self.pop_ys[idx]

    def min_expr(self, i, population=False):
        xs = self.pop_xs if population else self.xs
        ys = self.pop_ys if population else self.ys
        A = self.beta*self.alphas[i] * xs.T @  xs
        b = self.beta*self.alphas[i] * xs.T @ ys
        return A, b

    def risk(self, theta, relative=False, population=False):
        xs = self.pop_xs if population else self.xs
        ys = self.pop_ys if population else self.ys
        N = self.pop_N if population else self.N
        risk = np.linalg.norm(xs @ theta - ys)**2/N
        if not relative:
            return risk
        min_risk = np.linalg.norm(xs @ self.phi - ys)**2/self.N
        return risk/min_risk


class SampledMultinomialSubPop(SubPop):
    def __init__(self, xs, ys, *args,  **kwargs):
        self.xs = xs
        self.ys = ys
        self.N = self.xs.shape[0]
        phi_emp = LinearRegression(
            fit_intercept=False).fit(self.xs, self.ys).coef_
        super(SampledMultinomialSubPop, self).__init__(
            phi_emp, *args, **kwargs)
        self.kind = 'sampled'
        self.pop_N = self.N
        self.pop_xs = self.xs
        self.pop_ys = self.ys

    def sample(self, sample_size=None):
        lists = split_list_using_multinomial(
            np.array(range(self.pop_N)), self.alphas, sample_size=sample_size)
        self.xs = [self.pop_xs[i] for i in lists]
        self.ys = [self.pop_ys[i] for i in lists]

    def min_expr(self, i, population=False):
        xs = self.pop_xs if population else self.xs[i]
        ys = self.pop_ys if population else self.ys[i]
        if len(xs) == 0:
            return np.zeros((self.d, self.d)), np.zeros(self.d)
        A = self.beta*self.alphas[i] * xs.T @  xs
        b = self.beta*self.alphas[i] * xs.T @ ys
        return A, b

    def risk(self, theta, relative=False):
        xs = self.pop_xs
        ys = self.pop_ys
        N = self.pop_N
        risk = np.linalg.norm(xs @ theta - ys)**2/N
        if not relative:
            return risk
        min_risk = np.linalg.norm(xs @ self.phi - ys)**2/self.N
        return risk/min_risk


def generate_folktables_data():
    data_source = folktables.ACSDataSource(
        survey_year='2018', horizon='1-Year', survey='person')
    try:
        acs_data = data_source.get_data(states=["CA"])
    except FileNotFoundError:
        acs_data = data_source.get_data(states=["CA"], download=True)

    # process school
    def _p_schl(x):
        if x <= 15:
            return 0  # less than HS
        if x <= 19:
            return 1  # more than HS less than underrgraduate
        if x >= 20:
            return 2  # college and above
        return x

    # process means of transporation
    def _p_jwtr(x):
        if x == 1:
            return 0  # driving
        if x <= 7:
            return 1  # public transport
        if x in [8, 9, 10]:  # biked or walked
            return 2
        if x == 11:
            return 3  # worked from home
        if x == 12:
            return 4  # another method
        return x

    # process race variable
    def _p_rac1p(x):
        if x <= 4:
            return x-1
        if x == 5:  # combine Native and Alaska Native Only into one category
            # since Alaska Native Only contains just 9 individuals
            # and decision phi that achieves minimum risk is not defined
            return 3
        if x > 5:
            return x-2
        return x

    def postprocess(df, label_df, group_df):
        # Post process features
        df.loc[:, 'SCHL'] = df['SCHL'].apply(_p_schl)
        # process marital status (binary)
        df.loc[:, 'MAR'] = df['MAR'].apply(lambda x: 0 if x == 1 else 1)
        # process sex, dis, mig (binary, 0 index)
        df.loc[:, 'SEX'] = df['SEX'].apply(lambda x: x-1)
        df.loc[:, 'DIS'] = df['DIS'].apply(lambda x: x-1)
        df.loc[:, 'MIG'] = df['MIG'].apply(lambda x: x-1)
        # process means of transportation
        df.loc[:, 'JWTR'] = df['JWTR'].apply(_p_jwtr)
        categorical_features = ['SCHL', 'JWTR', 'MIG']
        for f in categorical_features:
            dummy_df = pd.get_dummies(
                df[f], prefix=f, prefix_sep="_", drop_first=True)
            df = pd.merge(
                left=df,
                right=dummy_df,
                left_index=True,
                right_index=True,
            )
        df.drop(labels=categorical_features+['RAC1P'], axis=1, inplace=True)
        # scale numerical variables for better conditioning
        df['POVPIP'] = df['POVPIP']/500
        df['AGEP'] = df['AGEP']/80

        # Post process label
        label_df.loc[:, 'JWMNP'] = label_df['JWMNP'].apply(
            lambda x: 10*np.log(1+x))

        # Post process group
        group_df.loc[:, 'RAC1P'] = group_df['RAC1P'].apply(_p_rac1p)
        return df, label_df, group_df

    ACSTravelTimeReg = folktables.BasicProblem(
        features=[
            'AGEP',  # see https://arxiv.org/pdf/2108.04884.pdf
            'SCHL',  # for detailed description of the columns and their values
            'MAR',
            'SEX',
            'DIS',
            'MIG',
            'RAC1P',
            'CIT',
            'JWTR',
            'POVPIP',
        ],
        target="JWMNP",

        group='RAC1P',
        preprocess=folktables.travel_time_filter,
    )
    features, label, group = postprocess(
        *ACSTravelTimeReg.df_to_pandas(acs_data))
    # replace nas with zeros in a dataframe
    features = features.fillna(0)
    # reolace boolean columns with integers
    features = features.astype(np.float32)
    return (features, label, group)


def split_list_using_multinomial(lst, probabilities, sample_size=None):
    """Splits a list using a multinomial distribution defined by the given probabilities."""
    # Make a copy of the list and shuffle it
    shuffled_lst = lst.copy()
    np.random.shuffle(shuffled_lst)
    if sample_size is not None:
        shuffled_lst = shuffled_lst[:sample_size]
    n = len(shuffled_lst)
    counts = np.random.multinomial(n, probabilities)

    sublists = []
    start = 0
    for count in counts:
        sublists.append(shuffled_lst[start:start+count])
        start += count

    return sublists
