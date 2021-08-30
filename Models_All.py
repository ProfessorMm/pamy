import torch
from torch import nn
import torch.optim as optim
from utils_updated import _image_standardization
import numpy as np


# LSTM implemented, layer number: 1
class LSTM_Model(nn.Module):

    def __init__(self, conv_head, time_step=16, input_size=128, hidden_size=64, output=3):  # the output of cnn_head decides the input_size

        super(LSTM_Model, self).__init__()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=0)  # weight_decay default 0.

        self.loss = nn.MSELoss(reduction='mean')
        self.exp_factor = 0.1
        self.num_params = 0

        self.conv_head = conv_head   # cnn layer before LSTM
        self.time_step = time_step
        self.input_size = input_size  # the number of expected features in input
        self.hidden_size = hidden_size  # the number of features in hidden state h, which should be the size of the output
        self.output = output
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)  # default: num_layers = 1
        self.linear = nn.Linear(self.hidden_size, self.output)  # linear(64,3)

    def forward(self, x):  # is it necessary to define h_0 and c_0 explicitly.

        batch_size = x.shape[0]  # because the batch size can be different for the last batch, not initialized in __init__
        x = self.conv_head(x)
        # after conv head, the tensor has the shape (N, T, F), batch_size * time sequence * features
        x, _ = self.lstm(x)
        # (h_0, c_0 default to zero if not provided), # (h_T, c_T) represent cell state/hidden state of last size

        # x has the shape (batch_size, time_step, hidden_size), use a fully connected layer to transfer to the output
        # dimension 3
        x = x.view(-1, self.hidden_size)
        x = self.linear(x)  # mapping hidden_size 64 to output dimension 3,  no activation function
        x = x.view(batch_size, self.time_step, self.output)
        return x

    # evaluate step by step, time_sequence = 1, for LSTM. the hidden state is (h, c), H_cell = H_h
    def evaluate_on_single_sequence(self, x, hidden_state=None):
        x = self.conv_head(x)  # shape (BatchSize, Time_Sequence, ExtractedFeatures)
        if hidden_state is None:
            hidden_state = (torch.zeros((1, x.shape[0], self.hidden_size),
                                        device=x.device),
                            torch.zeros((1, x.shape[0], self.hidden_size),
                                        device=x.device))  # h_0 and c_0 has the same shape, hidden_state = (h0,c0)
        result, hidden_state = self.lstm(x,
                                         hidden_state)  # there will be a update, x: (N,L,H_out).
        # in this case, N=1, L=1
        result = result.view(-1, self.hidden_size)
        result = self.linear(result)  # map the hidden out to the action pair
        return result, hidden_state

    # in use when doing online test
    def reset(self):
        return np.zeros(self.output)

    def criterion(self, a_imitator, a_exp):
        loss = self.loss(a_imitator, a_exp)
        return loss

    def weighted_criterion(self, a_imitator, a_exp):
        assert self.exp_factor >= 0
        weights = torch.exp(torch.abs(a_exp) * self.exp_factor)  # exp (|y|* alpha)
        error = a_imitator - a_exp
        return torch.sum(weights * torch.square(error)) / torch.sum(weights)

    # save the model after training
    def release(self, sdir):
        torch.save(self.state_dict(), sdir + "policy" + "_model.pth")
        torch.save(self.optimizer.state_dict(), sdir + "policy " + "_optim.pth")

    # load the trained model
    def load(self, ldir):
        try:
            self.load_state_dict(torch.load(ldir + "policy_model.pth"))
            self.optimizer.load_state_dict(torch.load(ldir + "policy_optim.pth"))
            print("load parameters are in" + ldir)
            return True
        except:
            print("parameters are not loaded")
            return False

    # count how many learnable parameters are there in the network
    def count_params(self):
        self.num_params = sum(param.numel() for param in self.parameters())
        return self.num_params