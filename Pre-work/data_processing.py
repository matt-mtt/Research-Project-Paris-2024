import numpy as np
import torch
import random
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class SemiContinuousSin:
    """
    Object for creating the semi-continuous sinusoid
    """
    def __init__(self, const_freq=False, const_amp=False, lower_bound=0, upper_bound=20*np.pi):
        self.const_freq = const_freq
        self.const_amp = const_amp
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.f_points = self.gen_f(const_freq, const_amp)
    
    def gen_noise(self, num_points):
        xx = self.f_points[:,0]
        yy = self.f_points[:,1]
        y_n = np.zeros(num_points)
        x_n_i = np.round(np.linspace(0, len(xx) - 1, num_points)).astype(int)
        # Normalize indices
        old_range = len(xx) - 0
        new_range = self.upper_bound - self.lower_bound
        x_n = x_n_i*new_range/old_range
        # Get true values and add noise
        for i in range(len(x_n)):
            temp_y = yy[x_n_i][i]
            if temp_y>0:
                temp_y += np.random.normal(loc=0, scale=0.02)
            else:
                temp_y += np.random.normal(loc=0, scale=0.04)
            y_n[i] = temp_y
        return np.column_stack((x_n,y_n)), yy[x_n_i]

    def gen_f(self, const_freq, const_amp):
        xx = np.linspace(start=self.lower_bound, stop=self.upper_bound, num=10000)
        yy = np.zeros(len(xx))
        if const_freq:
            if const_amp:
                for i in range(len(xx)):
                    yy[i] = np.sin(xx[i]) if np.sin(xx[i])>0 else 0
                return np.column_stack((xx,yy))
            else:
                # const_freq variable_amp
                zeroed = False
                rand_amp = random.uniform(0.98,1.02)
                for i in range(len(xx)):
                    val = np.sin(xx[i])*rand_amp
                    if zeroed and val>0:
                        zeroed = False
                        rand_amp = random.uniform(0.98,1.02)
                    if val < 0:
                        if zeroed == False:
                            zeroed = True
                        val = 0
                    yy[i] = val
                return np.column_stack((xx,yy))
        else:
            if const_amp:
                # variable freq const amp
                zeroed = False
                rand_width = random.uniform(0.98,1.02)
                for i in range(len(xx)):
                    val = np.sin(xx[i]*rand_width)
                    if zeroed and val>0:
                        zeroed = False
                        rand_width = random.uniform(0.98,1.02)
                    if val < 0:
                        if zeroed == False:
                            zeroed = True
                        val = 0
                    yy[i] = val
                return np.column_stack((xx,yy))
            else:
                # variable amp variable freq
                zeroed = False
                rand_width = random.uniform(0.98,1.02)
                rand_amp = random.uniform(0.98,1.02)
                for i in range(len(xx)):
                    val = np.sin(xx[i]*rand_width)*rand_amp
                    if zeroed and val>0:
                        zeroed = False
                        rand_width = random.uniform(0.98,1.02)
                        rand_amp = random.uniform(0.98,1.02)
                    if val < 0:
                        if zeroed == False:
                            zeroed = True
                        val = 0
                    yy[i] = val
                return np.column_stack((xx,yy))

def preprocess(noise, y, train_percent=50, logs=True):
    """
    Preprocess training sample to pass it into a LSTM pytorch model
    """
    mm = MinMaxScaler()
    ss = StandardScaler()
    train_len = int(noise.shape[0]*train_percent/100)
    # Reshape for tensor purposes
    y = y.reshape(-1,1)
    # Scale
    X_ss = ss.fit_transform(noise)
    y_mm = mm.fit_transform(y)
    # Create datasets
    X_train = X_ss[:train_len,:]
    y_train = y_mm[:train_len,:]
    X_test = X_ss[train_len:,:]
    y_test = y_mm[train_len:,:]
    # Create tensors
    X_train_t = Variable(torch.Tensor(X_train))
    y_train_t = Variable(torch.Tensor(y_train))
    X_test_t = Variable(torch.Tensor(X_test))
    y_test_t = Variable(torch.Tensor(y_test))
    # Reshape for LSTM
    X_train_t = torch.reshape(X_train_t, (X_train_t.shape[0], 
                                1, X_train_t.shape[1]))
    X_test_t = torch.reshape(X_test_t, (X_test_t.shape[0], 
                                1, X_test_t.shape[1]))
    if logs:
        print("Training Shape", X_train_t.shape, y_train_t.shape)
        print("Testing Shape", X_test_t.shape, y_test_t.shape)
    
    return X_train_t, y_train_t, X_test_t, y_test_t, mm, ss