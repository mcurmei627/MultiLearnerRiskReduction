import numpy as np
import folktables

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
        """Updates the alloations of the subpopulation

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
        alphas[i] = self.alphas[i]/2
        self.alphas = np.append(alphas, alphas[i])
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
        return(np.linalg.norm(np.dot(self.cov, theta - self.phi))**2 + self.sigmasq)



class EmpiricalSubPop(SubPop):
    def __init__(self, xs, ys, *args,  **kwargs):
        self.xs = xs
        self.ys = ys
        self.N = self.xs.shape[0]
        phi_emp = np.dot(np.linalg.pinv(self.xs), self.ys).flatten()
        super(EmpiricalSubPop, self).__init__(phi_emp, *args, **kwargs)
        self.kind = 'empirical'

    def min_expr(self, i):
        A = self.beta*self.alphas[i] * self.xs.T @  self.xs
        b = self.beta*self.alphas[i] *  self.xs.T @ self.ys
        return A, b
    
    def risk(self, theta):
        return np.linalg.norm(self.xs @ theta - self.ys)**2/self.N

def generate_folktables_subpops(alphas):
    data_source = folktables.ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"]) #, download=True)
    ACSTravelTimeReg = folktables.BasicProblem(
        features=[
            'AGEP',
            'SCHL',
            'MAR',
            'SEX',
            'DIS',
            'ESP',
            'MIG',
            'RELP',
            'RAC1P',
            'PUMA',
            'ST',
            'CIT',
            'OCCP',
            'JWTR',
            'POWPUMA',
            'POVPIP',
        ],
        target="JWMNP",
        target_transform=lambda x: x,
        group='RAC1P',
        preprocess=folktables.travel_time_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    features, label, group = ACSTravelTimeReg.df_to_numpy(acs_data)
    not_nan_inds = np.logical_not(np.isnan(label))
    features = features[not_nan_inds]
    label = label[not_nan_inds]
    group = group[not_nan_inds]
    label = 10*np.log(1+label)


    subpops = []
    for i,g in enumerate(np.unique(group)):
        g_inds = group==g
        g_beta = np.sum(g_inds) / len(group)

        subpops.append(
            EmpiricalSubPop(features[g_inds], label[g_inds], g_beta, alphas[i])
            )
    return subpops