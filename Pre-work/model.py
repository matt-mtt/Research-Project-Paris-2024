import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import copy

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super(RNN, self).__init__()
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #rnn
        self.fc = nn.Linear(hidden_size, 1) #Dense
        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        # Propagate input through LSTM
        output, hn = self.rnn(x, h_0) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

def train_model(cell, X_train, y_train, X_test, y_test, num_epochs=1500, lr=0.001, 
                input_size=2, hidden_size=5, lstm_layers=1):
    if cell == "LSTM":
        model = LSTM(input_size=input_size, hidden_size=hidden_size,
                      num_layers=lstm_layers, seq_length=X_train.shape[1])
    else:
        model = RNN(input_size=input_size, hidden_size=hidden_size,
                      num_layers=lstm_layers, seq_length=X_train.shape[1])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    min_val_metric = np.inf
    for e in range(num_epochs):
        outputs = model.forward(X_train)
        optimizer.zero_grad() # Zero the gradient at every new epoch
        
        loss = criterion(outputs, y_train)
        loss.backward()

        optimizer.step() # Backpropagation
        if e%100 == 0:
            eval_metric = evaluate(model, X_test, y_test)
            print(f"Epoch: {e}, loss: {loss.item():1.5f}, val MAE: {eval_metric:1.5f}")
            # Keep best model
            if eval_metric < min_val_metric:
                min_val_metric = eval_metric
                best_model = copy.deepcopy(model)
    
    return best_model

def evaluate(model, X_test, y_test, metric=torch.nn.L1Loss()):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        eval_ = metric(preds, y_test)
    model.train()
    return eval_

def predict(X_ss, y_mm, model):
    pred = model(X_ss) # Forward pass
    np_pred = pred.data.numpy() # Numpy conversion
    eval_metric = evaluate(model, X_ss, y_mm)
    return np_pred, eval_metric