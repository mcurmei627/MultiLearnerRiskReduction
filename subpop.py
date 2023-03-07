import numpy as np
#import folktables

class SubPop():
    def __init__(self, phi, beta, alphas, cov=None, sigmasq=1, price_sensitivity=0):
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
        self.price_sensitivity = price_sensitivity
        self.beta = beta
        self.alphas = alphas
        self.t = 0
        self.converged = False
        self.converged_for = 0
    
    def update_alpha(self, thetas, epsilon=0.1, prices = None):
        """Updates the allocations of the subpopulation

        Args:
            thetas (iterable of np.array): the decisions of each learner
            epsilon (float, optional): constant on the MWUD. Defaults to 0.1.
            prices (iterable of floats, optional): prices of each learner. Defaults to None.
        """
        num_learners = len(thetas)
        if prices is None:
            prices = np.zeros(num_learners)        
        r = [self.risk(theta, prices[i]) for i, theta in enumerate(thetas)]
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

    def risk(self, theta, price):
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
    
    def risk(self, theta, price = 0):
        """Computes the average risk for the subpopulation assumong that is  quadradic

        Args:
            theta (np.array): decision of a learner
        """        
        risk = np.linalg.norm(np.dot(self.cov, theta - self.phi))**2 + self.sigmasq + self.price_sensitivity*price
        #print('risk', risk)
        return(risk)



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
    
    def risk(self, theta, price=0, relative = False):
        risk = np.linalg.norm(self.xs @ theta - self.ys)**2/self.N + self.price_sensitivity*price
        if not relative:
            return risk
        min_risk = np.linalg.norm(self.xs @ self.phi - self.ys)**2/self.N + self.price_sensitivity*price
        return risk/min_risk

def generate_folktables_data(alphas):
    data_source = folktables.ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"]) #,download=True)
    # process school
    def _p_schl(x):
        if x<=15:
            return 0 # less than HS
        if x<=19:
            return 1 # more than HS less than underrgraduate
        if x>=20:
            return 2 # college and above
        return x

    # process means of transporation
    def _p_jwtr(x):
        if x == 1:
            return 0 # driving
        if x <= 7:
            return 1 # public transport
        if x in [8, 9, 10]: # biked or walked
            return 2
        if x == 11:
            return 3 # worked from home
        if x == 12:
            return 4 # another method
        return x

    # process race variable
    def _p_rac1p(x):
        if x <= 4:
            return x-1
        if x == 5: # combine Native and Alaska Native Only into one category
                # since Alaska Native Only contains just 9 individuals
                # and decision phi that achieves minimum risk is not defined
            return 3
        if x > 5:
            return x-2
        return x

    def postprocess(df, label_df, group_df):
        ## Post process features
        df.loc[:, 'SCHL'] = df['SCHL'].apply(_p_schl)
        # process marital status (binary) 
        df.loc[:, 'MAR'] = df['MAR'].apply(lambda x: 0 if x == 1 else 1)
        # process sex, dis, mig (binary, 0 index)
        df.loc[:, 'SEX'] = df['SEX'].apply(lambda x: x-1)
        df.loc[:, 'DIS'] = df['DIS'].apply(lambda x: x-1)
        df.loc[:, 'MIG'] = df['MIG'].apply(lambda x: x-1)
        # procee means of transportation
        df.loc[:, 'JWTR'] = df['JWTR'].apply(_p_jwtr)
        # process citizenship (0 indec)
        df.loc[:, 'CIT'] = df['CIT'].apply(lambda x:x-1)
        categorical_features = ['SCHL', 'JWTR', 'MIG', 'CIT']
        for f in categorical_features:
            dummy_df = pd.get_dummies(df[f], prefix = f, prefix_sep = "_")
            df = pd.merge(
                left = df,
                right = dummy_df,
                left_index = True,
                right_index = True,
            )
        df.drop(labels = categorical_features+['RAC1P'], axis=1, inplace=True)
        # scale numerical variables for better conditioning
        df['POVPIP'] = df['POVPIP']/500
        df['AGEP'] = df['AGEP']/80
        
        ## Post process label
        label_df.loc[:,'JWMNP'] = label_df['JWMNP'].apply(lambda x: 10*np.log(1+x))
        
        ## Post process group
        group_df.loc[:, 'RAC1P'] = group_df['RAC1P'].apply(_p_rac1p)
        return df, label_df, group_df

    ACSTravelTimeReg = folktables.BasicProblem(
        features=[
            'AGEP', # see https://arxiv.org/pdf/2108.04884.pdf
            'SCHL', # for detailed description of the columns and their values
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
        preprocess= folktables.travel_time_filter,
    )
    features, label, group = postprocess(*ACSTravelTimeReg.df_to_pandas(acs_data))
    return(features, label, group)