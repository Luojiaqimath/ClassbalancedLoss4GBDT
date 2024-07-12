from py_boost.gpu.losses import Loss, BCEMetric
import cupy as cp
import numpy as np


class CupyACELoss(Loss):
    def __init__(self, m, clip_value=1e-7):
        self.m = m
        self.clip_value = clip_value

    def base_score(self, y_true):
        means = cp.clip(y_true.mean(axis=0), self.clip_value, 1 - self.clip_value)
        return cp.log(means / (1 - means))

    def get_grad_hess(self, y_true, y_pred):
        pred = 1 / (1 + cp.exp(-y_pred))
        prob = cp.clip(pred, self.clip_value, 1 - self.clip_value)
        prob_m = 1-cp.maximum(prob-self.m, self.clip_value)
        grad = y_true*self.grad(prob)-(1-y_true)*self.grad(prob_m)
        hess = y_true*self.hess(prob)+(1-y_true)*self.hess(prob_m)
        return grad, hess

    def postprocess_output(self, y_pred):
        xp = np if type(y_pred) is np.ndarray else cp
        pred = 1 / (1 + xp.exp(-y_pred))
        pred = xp.clip(pred, self.clip_value, 1 - self.clip_value)

        return pred

    def get_metric_from_string(self, name=None):
        return BCEMetric()

    def grad(self, p):
        return p-1

    def hess(self, p):  
        return p*(1-p)
    
    
class CupyASLLoss(Loss):
    def __init__(self, r1, r2, m, clip_value=1e-7):
        self.r1 = r1
        self.r2 = r2
        self.m = m
        self.clip_value = clip_value

    def base_score(self, y_true):
        means = cp.clip(y_true.mean(axis=0), self.clip_value, 1 - self.clip_value)
        return cp.log(means / (1 - means))

    def get_grad_hess(self, y_true, y_pred):
        pred = 1 / (1 + cp.exp(-y_pred))
        prob = cp.clip(pred, self.clip_value, 1 - self.clip_value)
        prob_m = 1-cp.maximum(prob-self.m, self.clip_value)
        grad = y_true*self.grad(prob, self.r1)-(1-y_true)*self.grad(prob_m, self.r2)
        hess = y_true*(self.hess1(prob, self.r1)+self.hess2(prob, self.r1))+\
            (1-y_true)*(self.hess1(prob_m, self.r2)+self.hess2(prob_m, self.r2))
        return grad, hess

    def postprocess_output(self, y_pred):
        xp = np if type(y_pred) is np.ndarray else cp
        pred = 1 / (1 + xp.exp(-y_pred))
        pred = xp.clip(pred, self.clip_value, 1 - self.clip_value)

        return pred

    def get_metric_from_string(self, name=None):
        return BCEMetric()

    def grad(self, p, r):
        return ((1-p)**r)*(r*p*cp.log(p)+p-1)

    def hess1(self, p, r):  # dldp2*dp**2
        return ((1-p)**r)*(-r*(r-1)*p**2*cp.log(p)-2*r*p*(p-1)+(p-1)**2)

    def hess2(self, p, r):  # dldp*dp2 
        return ((1-p)**r)*(1-2*p)*(r*p*cp.log(p)+p-1)


class CupyAWELoss(Loss):
    def __init__(self, w, m, clip_value=1e-7):
        self.w = w
        self.m = m
        self.clip_value = clip_value

    def base_score(self, y_true):
        means = cp.clip(y_true.mean(axis=0), self.clip_value, 1 - self.clip_value)
        return cp.log(means / (1 - means))

    def get_grad_hess(self, y_true, y_pred):
        pred = 1 / (1 + cp.exp(-y_pred))
        prob = cp.clip(pred, self.clip_value, 1 - self.clip_value)
        prob_m = 1-cp.maximum(prob-self.m, self.clip_value)
        grad = self.w*y_true*self.grad(prob)-(1-y_true)*self.grad(prob_m)
        hess = self.w*y_true*self.hess(prob)+(1-y_true)*self.hess(prob_m)
        return grad, hess

    def postprocess_output(self, y_pred):
        xp = np if type(y_pred) is np.ndarray else cp
        pred = 1 / (1 + xp.exp(-y_pred))
        pred = xp.clip(pred, self.clip_value, 1 - self.clip_value)

        return pred

    def get_metric_from_string(self, name=None):
        return BCEMetric()

    def grad(self, p):
        return p-1

    def hess(self, p):  
        return p*(1-p)


class CupyFLLoss(Loss):
    def __init__(self, r, clip_value=1e-7):
        self.r = r
        self.clip_value = clip_value

    def base_score(self, y_true):
        means = cp.clip(y_true.mean(axis=0), self.clip_value, 1 - self.clip_value)
        return cp.log(means / (1 - means))

    def get_grad_hess(self, y_true, y_pred):
        pred = 1 / (1 + cp.exp(-y_pred))
        prob = cp.clip(pred, self.clip_value, 1 - self.clip_value)
        pt = (2*y_true-1)*(prob-0.5)+0.5
        grad = (2*y_true-1)*self.grad(pt)
        hess = self.hess1(pt)+self.hess2(pt)
        return grad, hess

    def postprocess_output(self, y_pred):
        xp = np if type(y_pred) is np.ndarray else cp
        pred = 1 / (1 + xp.exp(-y_pred))
        pred = xp.clip(pred, self.clip_value, 1 - self.clip_value)
        return pred

    def get_metric_from_string(self, name=None):
        return BCEMetric()

    def grad(self, p):
        return ((1-p)**self.r)*(self.r*p*cp.log(p)+p-1)

    def hess1(self, p):  # dldp2*dp**2
        return ((1-p)**self.r)*(-self.r*(self.r-1)*p**2*cp.log(p)-\
            2*self.r*p*(p-1)+(p-1)**2)
    
    def hess2(self, p):  # dldp*dp2 
        return ((1-p)**self.r)*(1-2*p)*(self.r*p*cp.log(p)+p-1)


class CupyWCELoss(Loss):
    def __init__(self, w, clip_value=1e-7):
        self.w = w
        self.clip_value = clip_value

    def base_score(self, y_true):
        means = cp.clip(y_true.mean(axis=0), self.clip_value, 1 - self.clip_value)
        return cp.log(means / (1 - means))

    def get_grad_hess(self, y_true, y_pred):
        pred = 1 / (1 + cp.exp(-y_pred))
        prob = cp.clip(pred, self.clip_value, 1 - self.clip_value)
        grad = self.w*y_true*self.grad(prob)-(1-y_true)*self.grad(1-prob)
        hess = self.w*y_true*self.hess(prob)+(1-y_true)*self.hess(1-prob)
        return grad, hess

    def postprocess_output(self, y_pred):
        xp = np if type(y_pred) is np.ndarray else cp
        pred = 1 / (1 + xp.exp(-y_pred))
        pred = xp.clip(pred, self.clip_value, 1 - self.clip_value)

        return pred

    def get_metric_from_string(self, name=None):
        return BCEMetric()

    def grad(self, p):
        return p-1

    def hess(self, p):  
        return p*(1-p)
    
    
class CupyCBELoss(Loss):
    def __init__(self, b, clip_value=1e-7):
        self.b = b
        self.clip_value = clip_value

    def base_score(self, y_true):
        means = cp.clip(y_true.mean(axis=0), self.clip_value, 1 - self.clip_value)
        return cp.log(means / (1 - means))

    def get_grad_hess(self, y_true, y_pred):
        pred = 1 / (1 + cp.exp(-y_pred))
        prob = cp.clip(pred, self.clip_value, 1 - self.clip_value)
        n_pos = cp.sum(y_true)
        n_neg = len(y_true)-n_pos
        grad = (1-self.b)/(1-self.b**n_pos)*y_true*self.grad(prob)-(1-self.b)/(1-self.b**n_neg)*(1-y_true)*self.grad(1-prob)
        hess = (1-self.b)/(1-self.b**n_pos)*y_true*self.hess(prob)+(1-self.b)/(1-self.b**n_neg)*(1-y_true)*self.hess(1-prob)
        return grad, hess

    def postprocess_output(self, y_pred):
        xp = np if type(y_pred) is np.ndarray else cp
        pred = 1 / (1 + xp.exp(-y_pred))
        pred = xp.clip(pred, self.clip_value, 1 - self.clip_value)

        return pred

    def get_metric_from_string(self, name=None):
        return BCEMetric()

    def grad(self, p):
        return p-1

    def hess(self, p):  
        return p*(1-p)
    