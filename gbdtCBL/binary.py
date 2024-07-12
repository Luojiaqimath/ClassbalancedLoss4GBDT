import numpy as np


class ACELoss:
    def __init__(self, m, clip_value=1e-7):
        self.m = m
        self.clip_value = clip_value

    def __call__(self, y_true, preds):
        pred = 1 / (1 + np.exp(-preds))
        prob = np.clip(pred, self.clip_value, 1 - self.clip_value)
        prob_m = 1 - np.maximum(prob - self.m, self.clip_value)
        grad = y_true * self.grad(prob) - (1 - y_true) * self.grad(prob_m)
        hess = y_true * self.hess(prob) + (1 - y_true) * self.hess(prob_m)
        return grad, hess

    def grad(self, p):
        return p - 1

    def hess(self, p):
        return p * (1 - p)


class ASLLoss:
    def __init__(self, r1, r2, m, clip_value=1e-7):
        self.r1 = r1
        self.r2 = r2
        self.m = m
        self.clip_value = clip_value

    def __call__(self, y_true, preds):
        pred = 1 / (1 + np.exp(-preds))
        prob = np.clip(pred, self.clip_value, 1 - self.clip_value)
        prob_m = 1 - np.maximum(prob - self.m, self.clip_value)
        grad = y_true * self.grad(prob, self.r1) - (1 - y_true) * self.grad(prob_m, self.r2)
        hess = y_true * (self.hess1(prob, self.r1) + self.hess2(prob, self.r1)) + \
               (1 - y_true) * (self.hess1(prob_m, self.r2) + self.hess2(prob_m, self.r2))
        return grad, hess

    def grad(self, p, r):
        return ((1 - p) ** r) * (r * p * np.log(p) + p - 1)

    def hess1(self, p, r):  # dldp2*dp**2
        return ((1 - p) ** r) * (-r * (r - 1) * p ** 2 * np.log(p) - 2 * r * p * (p - 1) + (p - 1) ** 2)

    def hess2(self, p, r):  # dldp*dp2
        return ((1 - p) ** r) * (1 - 2 * p) * (r * p * np.log(p) + p - 1)



class AWELoss():
    def __init__(self, r1, m, clip_value=1e-7):
        self.r1 = r1
        self.m = m
        self.clip_value = clip_value

    def __call__(self, y_true, preds):
        pred = 1 / (1 + np.exp(-preds))
        prob = np.clip(pred, self.clip_value, 1 - self.clip_value)
        prob_m = 1-np.maximum(prob-self.m, self.clip_value)
        grad = self.r1*y_true*self.grad(prob)-(1-y_true)*self.grad(prob_m)
        hess = self.r1*y_true*self.hess(prob)+(1-y_true)*self.hess(prob_m)
        return grad, hess

    def grad(self, p):
        return p-1

    def hess(self, p):  
        return p*(1-p)
    

class FLLoss:
    def __init__(self, r1, clip_value=1e-7):
        self.r1 = r1
        self.clip_value = clip_value

    def __call__(self, y_true, preds):
        pred = 1 / (1 + np.exp(-preds))
        prob = np.clip(pred, self.clip_value, 1 - self.clip_value)
        grad = y_true * self.grad(prob, self.r1) - (1 - y_true) * self.grad(1 - prob, self.r1)
        hess = y_true * (self.hess1(prob, self.r1) + self.hess2(prob, self.r1)) + \
               (1 - y_true) * (self.hess1(1 - prob, self.r1) + self.hess2(1 - prob, self.r1))
        return grad, hess

    def grad(self, p, r):
        return ((1 - p) ** r) * (r * p * np.log(p) + p - 1)

    def hess1(self, p, r):  # dldp2*dp**2
        return ((1 - p) ** r) * (-r * (r - 1) * p ** 2 * np.log(p) - 2 * r * p * (p - 1) + (p - 1) ** 2)

    def hess2(self, p, r):  # dldp*dp2
        return ((1 - p) ** r) * (1 - 2 * p) * (r * p * np.log(p) + p - 1)
    
    
class WCELoss():
    def __init__(self, r1, clip_value=1e-7):
        self.r1 = r1
        self.clip_value = clip_value

    def __call__(self, y_true, preds):
        pred = 1 / (1 + np.exp(-preds))
        prob = np.clip(pred, self.clip_value, 1 - self.clip_value)
        grad = self.r1*y_true*self.grad(prob)-(1-y_true)*self.grad(1-prob)
        hess = self.r1*y_true*(self.hess(prob))+(1-y_true)*(self.hess(1-prob))
        return grad, hess

    def grad(self, p):
        return p-1

    def hess(self, p):  
        return p*(1-p)


class CBELoss():
    def __init__(self, b, clip_value=1e-7):
        self.b = b
        self.clip_value = clip_value

    def __call__(self, y_true, preds):
        pred = 1 / (1 + np.exp(-preds))
        prob = np.clip(pred, self.clip_value, 1 - self.clip_value)
        n_pos = np.sum(y_true)
        n_neg = len(y_true)-n_pos
        grad = (1-self.b)/(1-self.b**n_pos)*y_true*self.grad(prob)-(1-self.b)/(1-self.b**n_neg)*(1-y_true)*self.grad(1-prob)
        hess = (1-self.b)/(1-self.b**n_pos)*y_true*(self.hess(prob))+(1-self.b)/(1-self.b**n_neg)*(1-y_true)*(self.hess(1-prob))
        return grad, hess

    def grad(self, p):
        return p-1

    def hess(self, p):  
        return p*(1-p)
    
    