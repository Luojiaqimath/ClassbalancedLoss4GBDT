import cupy as cp
import numpy as np
from py_boost.gpu.losses import Loss, CrossEntropyMetric
from sklearn.preprocessing import OneHotEncoder
    


def softmax(x, clip_val=1e-5):
    
    xp = np if type(x) is np.ndarray else cp
    exp_p = xp.exp(x - x.max(axis=1, keepdims=True))

    return xp.clip(exp_p / exp_p.sum(axis=1, keepdims=True), clip_val, 1 - clip_val)


class CupyACEMulti(Loss):
    def __init__(self, m, clip_value=1e-7):
        self.m = m
        self.clip_value = clip_value

    def base_score(self, y_true):
        num_classes = int(y_true.max() + 1)
        hist = cp.zeros((num_classes,), dtype=cp.float32)
        return hist

    def get_grad_hess(self, y_true, y_pred):
        p = softmax(y_pred, self.clip_value)
        pm = cp.maximum(p-self.m, self.clip_value)
        encoder = OneHotEncoder()
        encoded_label = encoder.fit_transform(y_true.get().reshape(-1, 1)).toarray().astype(np.float32)
        encoded_label = cp.asarray(encoded_label)
        grad = -encoded_label*(encoded_label-p)-(1-encoded_label)*(encoded_label-pm)
        hess =  encoded_label*(encoded_label-p)*(encoded_label+p-1)+(1-encoded_label)*(encoded_label-pm)*(encoded_label+pm-1)
        return grad, hess

    def postprocess_output(self, y_pred):
        return softmax(y_pred, self.clip_value)

    def preprocess_input(self, y_true):
        return y_true[:, 0].astype(cp.int32)

    def get_metric_from_string(self, name=None):
        return CrossEntropyMetric()
    

class CupyASLMulti(Loss):
    def __init__(self, r1, r2, m, clip_value=1e-7):
        self.r1 = r1
        self.r2 = r2
        self.m = m
        self.clip_value = clip_value

    def base_score(self, y_true):
        num_classes = int(y_true.max() + 1)
        hist = cp.zeros((num_classes,), dtype=cp.float32)
        return hist

    def get_grad_hess(self, y_true, y_pred):
        p = softmax(y_pred, self.clip_value)
        pm = cp.maximum(p-self.m, self.clip_value)
        encoder = OneHotEncoder()
        encoded_label = encoder.fit_transform(y_true.get().reshape(-1, 1)).toarray().astype(np.float32)
        encoded_label = cp.asarray(encoded_label)
        
        pt = cp.sum(p*encoded_label, axis=1, keepdims=True)
        
        grad = encoded_label*self.dldpXp(pt, self.r1)*(encoded_label-p)+\
            (1-encoded_label)*self.dldpXp(pt, self.r2)*(encoded_label-pm)
        hess = encoded_label*(self.dldp2Xp2(pt, self.r1)*(encoded_label-p)**2+\
            self.dldpXp(pt, self.r1)*(encoded_label-p)*(1-2*p))+\
                (1-encoded_label)*(self.dldp2Xp2(pt, self.r2)*(encoded_label-pm)**2+\
            self.dldpXp(pt, self.r2)*(encoded_label-pm)*(1-2*pm))

        return grad, hess
    
    def dldpXp(self, p, r):  # dldp*p
        result = cp.zeros(p.shape, dtype=cp.float32)
        p1 = p[(0<p)&(p<1)] 
        result[(0<p)&(p<1)] = (1-p1)**(r-1)*(r*p1*cp.log(p1)+p1-1)
        return result
        
    def dldp2Xp2(self, p, r):  # dldp2*p**2
        result = cp.zeros(p.shape, dtype=cp.float32)
        p1 = p[(0<p)&(p<1)]
        result[(0<p)&(p<1)] = -(1-p1)**(r-2)*(r*(r-1)*p1**2*cp.log(p1)+\
            2*r*p1*(p1-1)-(p1-1)**2)
        return result
    
    def postprocess_output(self, y_pred):
        return softmax(y_pred, self.clip_value)

    def preprocess_input(self, y_true):
        return y_true[:, 0].astype(cp.int32)

    def get_metric_from_string(self, name=None):
        return CrossEntropyMetric()
    
    
class CupyAWEMulti(Loss):
    def __init__(self, w, m, clip_value=1e-7):
        self.w = w
        self.m = m
        self.clip_value = clip_value

    def base_score(self, y_true):
        num_classes = int(y_true.max() + 1)
        hist = cp.zeros((num_classes,), dtype=cp.float32)
        return hist

    def get_grad_hess(self, y_true, y_pred):
        p = softmax(y_pred, self.clip_value)
        pm = cp.maximum(p-self.m, self.clip_value)
        encoder = OneHotEncoder()
        encoded_label = encoder.fit_transform(y_true.get().reshape(-1, 1)).toarray().astype(np.float32)
        encoded_label = cp.asarray(encoded_label)

        grad = -self.w*encoded_label*(encoded_label-p)-(1-encoded_label)*(encoded_label-pm)
        hess =  self.w*encoded_label*(encoded_label-p)*(encoded_label+p-1)+(1-encoded_label)*(encoded_label-pm)*(encoded_label+pm-1)
        return grad, hess

    def postprocess_output(self, y_pred):
        return softmax(y_pred, self.clip_value)

    def preprocess_input(self, y_true):
        return y_true[:, 0].astype(cp.int32)

    def get_metric_from_string(self, name=None):
        return CrossEntropyMetric()
    

class CupyFLMulti(Loss):

    def __init__(self, r, clip_value=1e-7):
        self.r = r
        self.clip_value = clip_value

    def base_score(self, y_true):
        num_classes = int(y_true.max() + 1)
        hist = cp.zeros((num_classes,), dtype=cp.float32)
        return hist

    def get_grad_hess(self, y_true, y_pred):
        p = softmax(y_pred, self.clip_value)
        encoder = OneHotEncoder()
        encoded_label = encoder.fit_transform(y_true.get().reshape(-1, 1)).toarray().astype(np.float32)
        encoded_label = cp.asarray(encoded_label)
        
        pt = cp.sum(p*encoded_label, axis=1, keepdims=True)
        
        grad = self.dldpXp(pt)*(encoded_label-p)
        hess = self.dldp2Xp2(pt)*(encoded_label-p)**2+\
            self.dldpXp(pt)*(encoded_label-p)*(1-2*p)
        return grad, hess
    
    def dldpXp(self, p):  # dldp*p
        result = cp.zeros(p.shape, dtype=cp.float32)
        p1 = p[(0<p)&(p<1)] 
        result[(0<p)&(p<1)] = (1-p1)**(self.r-1)*(self.r*p1*cp.log(p1)+p1-1)
        return result
        
    def dldp2Xp2(self, p):  # dldp2*p**2
        result = cp.zeros(p.shape, dtype=cp.float32)
        p1 = p[(0<p)&(p<1)]
        result[(0<p)&(p<1)] = -(1-p1)**(self.r-2)*(self.r*(self.r-1)*p1**2*cp.log(p1)+\
            2*self.r*p1*(p1-1)-(p1-1)**2)
        return result
    
    def postprocess_output(self, y_pred):
        return softmax(y_pred, self.clip_value)

    def preprocess_input(self, y_true):
        return y_true[:, 0].astype(cp.int32)

    def get_metric_from_string(self, name=None):
        return CrossEntropyMetric()
    
       
class CupyWCEMulti(Loss):
    def __init__(self, w, clip_value=1e-7):
        self.w = w
        self.clip_value = clip_value

    def base_score(self, y_true):
        num_classes = int(y_true.max() + 1)
        hist = cp.zeros((num_classes,), dtype=cp.float32)
        return hist

    def get_grad_hess(self, y_true, y_pred):
        p = softmax(y_pred, self.clip_value)
        encoder = OneHotEncoder()
        encoded_label = encoder.fit_transform(y_true.get().reshape(-1, 1)).toarray().astype(np.float32)
        encoded_label = cp.asarray(encoded_label)

        grad = -self.w*encoded_label*(encoded_label-p)-(1-encoded_label)*(encoded_label-p)
        hess =  self.w*encoded_label*(encoded_label-p)*(encoded_label+p-1)+(1-encoded_label)*(encoded_label-p)*(encoded_label+p-1)
        return grad, hess

    def postprocess_output(self, y_pred):
        return softmax(y_pred, self.clip_value)

    def preprocess_input(self, y_true):
        return y_true[:, 0].astype(cp.int32)

    def get_metric_from_string(self, name=None):
        return CrossEntropyMetric()
    

class CBELoss(Loss):
    def __init__(self, b, clip_value=1e-7):
        self.b = b
        self.clip_value = clip_value

    def base_score(self, y_true):
        num_classes = int(y_true.max() + 1)
        hist = cp.zeros((num_classes,), dtype=cp.float32)
        return hist

    def get_grad_hess(self, y_true, y_pred):
        p = softmax(y_pred, self.clip_value)
        encoder = OneHotEncoder()
        encoded_label = encoder.fit_transform(y_true.get().reshape(-1, 1)).toarray().astype(np.float32)
        encoded_label = cp.asarray(encoded_label)
        
        n_pos = cp.sum(y_true, axis=0)
        ratio = (1-self.b)/(1-self.b**n_pos)
        
        grad = p-encoded_label
        hess = (encoded_label-p)*(encoded_label+p-1)
        
        grad *= ratio
        hess *= ratio
        return grad, hess

    def postprocess_output(self, y_pred):
        return softmax(y_pred, self.clip_value)

    def preprocess_input(self, y_true):
        return y_true[:, 0].astype(cp.int32)

    def get_metric_from_string(self, name=None):
        return CrossEntropyMetric()
    