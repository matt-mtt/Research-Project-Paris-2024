from pyexpat.errors import XML_ERROR_INCORRECT_ENCODING
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torchvision.models as models

# ------------- Models without temporal dimension -------------- #

class SimpleCNN(nn.Module):
    def __init__(self, numChannels, classes = 8):
		# call the parent constructor
        super(SimpleCNN, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=20,
            kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
            kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=24200, out_features=5000)
        self.relu3 = nn.ReLU()
        # initialize our linear output
        self.fc2 = nn.Linear(in_features=5000, out_features=classes)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        # first set of CONV => RELU => POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # second set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it through FC
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        x = self.fc2(x)
        output = self.relu4(x)
        # return the output predictions
        return output


class CNN(nn.Module):
    def __init__(self, train_CNN=False):
        super(CNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, 64)
        self.reg = nn.Linear(64,8)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)

    def forward(self, images):
        features = self.inception(images)
        dropout = self.dropout(features)
        out_reg = self.tanh(self.reg(dropout))
        return out_reg
    
# ---------- End of models without temporal dimension ---------- #

# --------------- Models with temporal dimension --------------- #

class CNNLSTM(nn.Module):
    def __init__(self, numChannels, classes = 12):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=16,
            kernel_size=(3, 3))
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64,
            kernel_size=(3, 3))
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=(3, 3))
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.avgpool = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.fc_cnn1 = nn.Linear(in_features=25344, out_features=256)
        self.batchnorm3 = nn.BatchNorm1d(256)
        # self.fc_cnn2 = nn.Linear(in_features=5000, out_features=300)
        self.lstm = nn.LSTM(input_size=256, hidden_size=64, num_layers=1, bidirectional=True)
        self.fc_lstm = nn.Linear(in_features=128, out_features=classes)
        self.relu = nn.ReLU()
        

    def cnn_forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.batchnorm2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.avgpool(x)
        # x = self.batchnorm3(x)
        x = torch.flatten(x, 1)
        x = self.fc_cnn1(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        # x = self.fc_cnn2(x)
        # x = self.relu(x)
        return x

    def forward(self, temporal_x):
        """
        temporal_x is a 5D vector : [B,S,C,W,H]
        Where:
            - B is the batch size
            - S is the lenght of a temporal sequence
            - C is the number of channels
            - W is the width of an image
            - H is the height of an image
        """
        h = None
        cnn_output = []
        # Iterate over temporal dimension
        for t in range(temporal_x.size(1)):
            with torch.no_grad():
                x = self.cnn_forward(temporal_x[:,t,:,:,:])
                cnn_output.append(x)
        out, h = self.lstm(torch.stack(cnn_output), h)
        # out, h = self.lstm2(x)
        x = self.fc_lstm(out)
        x = self.relu(x)
        return x.permute(1,0,2)

# ------------ End of models with temporal dimension ----------- #

# ------------------------ Seq2seq model ----------------------- #

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input, hidden):
        # input = input.view(1,1,-1)
        output, hidden = self.lstm(input, hidden)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.relu(output)
        return output, hidden