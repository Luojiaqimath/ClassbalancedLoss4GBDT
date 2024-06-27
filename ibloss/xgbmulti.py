import numpy as np
from sklearn.preprocessing import OneHotEncoder


def softmax(x):
    kEps = 1e-7
    e = np.exp(x)
    return np.clip(e / np.expand_dims(np.sum(e, axis=1), axis=1), kEps, 1 - kEps)



class XGBACEMulti():
    def __init__(self, m):
        self.m = m

    def __call__(self, labels, preds):
        p = softmax(preds)
        pm = np.maximum(p-self.m, 1e-7)
        grad = np.zeros(preds.shape)
        hess = np.zeros(preds.shape)
        encoder = OneHotEncoder()
        encoded_label = encoder.fit_transform(labels.reshape(-1, 1)).toarray()

        grad = -encoded_label*(encoded_label-p)-(1-encoded_label)*(encoded_label-pm)
        hess =  encoded_label*(encoded_label-p)*(encoded_label+p-1)+(1-encoded_label)*(encoded_label-pm)*(encoded_label+pm-1)
        return grad.reshape(grad.shape[0]*grad.shape[1]), hess.reshape(hess.shape[0]*hess.shape[1])

    
    
class XGBASLMulti():
    def __init__(self, r1, r2, m):
        self.r1 = r1
        self.r2 = r2
        self.m = m

    def __call__(self, labels, preds):
        p = softmax(preds)
        pm = np.maximum(p-self.m, 1e-7)
        grad = np.zeros(preds.shape)
        hess = np.zeros(preds.shape)
        encoder = OneHotEncoder()
        encoded_label = encoder.fit_transform(labels.reshape(-1, 1)).toarray()
        
        pt = np.sum(p*encoded_label, axis=1, keepdims=True)
        
        grad = encoded_label*self.dldpXp(pt, self.r1)*(encoded_label-p)+\
            (1-encoded_label)*self.dldpXp(pt, self.r2)*(encoded_label-pm)
        hess = encoded_label*(self.dldp2Xp2(pt, self.r1)*(encoded_label-p)**2+\
            self.dldpXp(pt, self.r1)*(encoded_label-p)*(1-2*p))+\
                (1-encoded_label)*(self.dldp2Xp2(pt, self.r2)*(encoded_label-pm)**2+\
            self.dldpXp(pt, self.r2)*(encoded_label-pm)*(1-2*pm))

        return grad.reshape(grad.shape[0]*grad.shape[1]), hess.reshape(hess.shape[0]*hess.shape[1])
    
    def dldpXp(self, p, r):  # dldp*p
        result = np.zeros(p.shape)
        p1 = p[(0<p)&(p<1)] 
        result[(0<p)&(p<1)] = (1-p1)**(r-1)*(r*p1*np.log(p1)+p1-1)
        return result
        
    def dldp2Xp2(self, p, r):  # dldp2*p**2
        result = np.zeros(p.shape)
        p1 = p[(0<p)&(p<1)]
        result[(0<p)&(p<1)] = -(1-p1)**(r-2)*(r*(r-1)*p1**2*np.log(p1)+\
            2*r*p1*(p1-1)-(p1-1)**2)
        return result

    
 
class XGBAWEMulti():
    def __init__(self, r, m):
        self.r = r
        self.m = m

    def __call__(self, labels, preds):
        p = softmax(preds)
        pm = np.maximum(p-self.m, 1e-7)
        grad = np.zeros(preds.shape)
        hess = np.zeros(preds.shape)
        encoder = OneHotEncoder()
        encoded_label = encoder.fit_transform(labels.reshape(-1, 1)).toarray()

        grad = -self.r*encoded_label*(encoded_label-p)-(1-encoded_label)*(encoded_label-pm)
        hess =  self.r*encoded_label*(encoded_label-p)*(encoded_label+p-1)+(1-encoded_label)*(encoded_label-pm)*(encoded_label+pm-1)
        return grad.reshape(grad.shape[0]*grad.shape[1]), hess.reshape(hess.shape[0]*hess.shape[1])

  

class XGBFLMulti():
    def __init__(self, r):
        self.r = r

    def __call__(self, labels, preds):
        p = softmax(preds)
        grad = np.zeros(preds.shape)
        hess = np.zeros(preds.shape)
        encoder = OneHotEncoder()
        encoded_label = encoder.fit_transform(labels.reshape(-1, 1)).toarray()
        
        pt = np.sum(p*encoded_label, axis=1, keepdims=True)
        
        grad = self.dldpXp(pt)*(encoded_label-p)
        hess = self.dldp2Xp2(pt)*(encoded_label-p)**2+\
            self.dldpXp(pt)*(encoded_label-p)*(1-2*p)

        return grad.reshape(grad.shape[0]*grad.shape[1]), hess.reshape(hess.shape[0]*hess.shape[1])
    
    def dldpXp(self, p):  # dldp*p
        result = np.zeros(p.shape)
        p1 = p[(0<p)&(p<1)] 
        result[(0<p)&(p<1)] = (1-p1)**(self.r-1)*(self.r*p1*np.log(p1)+p1-1)
        return result
        
    def dldp2Xp2(self, p):  # dldp2*p**2
        result = np.zeros(p.shape)
        p1 = p[(0<p)&(p<1)]
        result[(0<p)&(p<1)] = -(1-p1)**(self.r-2)*(self.r*(self.r-1)*p1**2*np.log(p1)+\
            2*self.r*p1*(p1-1)-(p1-1)**2)
        return result

    
    
class XGBWCEMulti():
    def __init__(self, r):
        self.r = r

    def __call__(self, labels, preds):
        p = softmax(preds)
        grad = np.zeros(preds.shape)
        hess = np.zeros(preds.shape)
        encoder = OneHotEncoder()
        encoded_label = encoder.fit_transform(labels.reshape(-1, 1)).toarray()

        grad = -self.r*encoded_label*(encoded_label-p)-(1-encoded_label)*(encoded_label-p)
        hess =  self.r*encoded_label*(encoded_label-p)*(encoded_label+p-1)+(1-encoded_label)*(encoded_label-p)*(encoded_label+p-1)
        return grad.reshape(grad.shape[0]*grad.shape[1]), hess.reshape(hess.shape[0]*hess.shape[1])

