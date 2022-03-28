from scipy.special import logsumexp
from misc_utils import *



class GMM(object):
    def __init__(self, k, tol = 1e-3, reg_covar = 1e-7):
        self.K = k #这个参数是什么意思，嗷，K个类别
        self.tol = tol
        self.reg_covar=reg_covar
        self.times = 100
        self.loglike = 0


    def fit(self, trainMat):
        self.X = trainMat
        self.N, self.D = trainMat.shape
        self.GMM_EM()

    # gmm入口
    def GMM_EM(self):
        self.scale_data()   #数据预处理，归一化至[0，1]，每个维度上操作。
        self.init_params()  #初始化参数，均值为随机，协方差为0.1的对角矩阵，因为有k个高斯分布，所以有k个协方差，alpha初始化为1/K
        for i in range(self.times):
            log_prob_norm, self.gamma = self.e_step(self.X)  #gamma相当于过了一个softmax，得到的是变量属于哪一个高斯分布的概率
            self.mu, self.cov, self.alpha = self.m_step()
            newloglike = self.loglikelihood(log_prob_norm)
            # print(newloglike)
            if abs(newloglike - self.loglike) < self.tol:  #迭代值收敛
                break
            self.loglike = newloglike


    #预测类别
    def predict(self, testMat):
        log_prob_norm, gamma = self.e_step(testMat)
        category = gamma.argmax(axis=1).flatten().tolist()[0]
        return np.array(category)


    #e步，估计gamma，就是特征属于哪一个高斯分布的概率
    def e_step(self, data):
        gamma_log_prob = np.mat(np.zeros((self.N, self.K)))   #每一个样本属于第K个高斯分布的概率

        for k in range(self.K):   #计算在第K个高斯分布下，样本出现的对应概率
            gamma_log_prob[:, k] = log_weight_prob(data, self.alpha[k], self.mu[k], self.cov[k])

        log_prob_norm = logsumexp(gamma_log_prob, axis=1) #logsumexp是先对矩阵以e次方求和取对数
        log_gamma = gamma_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, np.exp(log_gamma)


    #m步，最大化loglikelihood
    def m_step(self):
        newmu = np.zeros([self.K, self.D])
        newcov = []
        newalpha = np.zeros(self.K)
        for k in range(self.K):
            Nk = np.sum(self.gamma[:, k])
            newmu[k, :] = np.dot(self.gamma[:, k].T, self.X) / Nk   #考虑特征属于哪一个高斯分布之后，计算均值，相当于一个加权平均
            cov_k = self.compute_cov(k, Nk) #和求均值一样的道理
            newcov.append(cov_k)
            newalpha[k] = Nk / self.N

        newcov = np.array(newcov)
        return newmu, newcov, newalpha


    #计算cov，防止非正定矩阵reg_covar
    def compute_cov(self, k, Nk):
        diff = np.mat(self.X - self.mu[k])
        cov = np.array(diff.T * np.multiply(diff, self.gamma[:,k]) / Nk)
        cov.flat[::self.D + 1] += self.reg_covar
        return cov


    #数据预处理，归一化至[0，1]，每个维度上操作
    def scale_data(self):
        for d in range(self.D):
            max_ = self.X[:, d].max()
            min_ = self.X[:, d].min()
            self.X[:, d] = (self.X[:, d] - min_) / (max_ - min_)
        self.xj_mean = np.mean(self.X, axis=0)
        self.xj_s = np.sqrt(np.var(self.X, axis=0))


    #初始化参数
    def init_params(self):
        self.mu = np.random.rand(self.K, self.D)
        self.cov = np.array([np.eye(self.D)] * self.K) * 0.1
        self.alpha = np.array([1.0 / self.K] * self.K)


    #log近似算法，可以防止underflow，overflow
    def loglikelihood(self, log_prob_norm):
        return np.sum(log_prob_norm)


    # def loglikelihood(self):
    #     P = np.zeros([self.N, self.K])
    #     for k in range(self.K):
    #         P[:,k] = prob(self.X, self.mu[k], self.cov[k])
    #
    #     return np.sum(np.log(P.dot(self.alpha)))

