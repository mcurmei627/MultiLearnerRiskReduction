import numpy as np

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
    def __init__(self, *args,  N=100, **kwargs):
        super(EmpiricalSubPop, self).__init__(*args, **kwargs)
        self.N = N
        self.kind = 'empirical'
        self.make_data()
        self.phi_emp = np.dot(np.linalg.pinv(self.xs), self.ys).flatten()


    def make_data(self):
        self.xs = np.random.multivariate_normal(np.zeros(self.d), self.cov, size=self.N)
        self.ys = np.array([np.dot(self.phi, x)+
                       np.random.normal(0, scale=np.sqrt(self.sigmasq)) for x in self.xs])

    def min_expr(self, i):
        A = self.beta*self.alphas[i] * self.xs
        b = self.beta*self.alphas[i] * self.ys
        return A, b
    
    def risk(self, theta):
        return np.linalg.norm(self.xs @ theta - self.ys)**2/self.N