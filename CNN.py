import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, channel_in=1, channel_out=1, dropout=0.0,
                 length_data=100, length_label=1, filter_size=21, width=4):
     
        super(CNN, self).__init__()

        l = length_data
        self.conv1 = nn.Sequential(  nn.Conv2d( in_channels=channel_in,
                                                out_channels=30 * channel_in,
                                                kernel_size=(filter_size, 3),
                                                stride=(1, 1),
                                                padding=(0, 1),
                                                bias=True),
                                     nn.Tanh()
                                  )
        l = (l - filter_size + 1)
        self.conv2 = nn.Sequential(  nn.Conv2d( in_channels=30 * channel_in,
                                                out_channels=10 * channel_in,
                                                kernel_size=(filter_size, 2),
                                                stride=(1, 1),
                                                padding=(0, 0),
                                                bias=True ),
                                     nn.Tanh()

                                   )
        l = (l - filter_size + 1)
        self.conv3 = nn.Sequential(  nn.Conv2d( in_channels=10 * channel_in,
                                                out_channels=3 * channel_in,
                                                kernel_size=(filter_size, 2),
                                                stride=(1, 1),
                                                padding=(0, 0),
                                                bias=True ),
                                     nn.Tanh()

                                   )
        l = (l - filter_size + 1)
        # self.conv4 = nn.Sequential(  nn.Conv2d( in_channels=10 * channel_in,
        #                                         out_channels=5 * channel_in,
        #                                         kernel_size=(filter_size, 1),
        #                                         stride=(1, 1),
        #                                         padding=(0, 0),
        #                                         bias=True ),
        #                              nn.Tanh()

        #                           )
        # l = (l - filter_size + 1)
        # self.conv5 = nn.Sequential( nn.Conv2d( in_channels=20 * channel_in,
        #                                        out_channels=10 * channel_in,
        #                                        kernel_size=(filter_size, 1),
        #                                        stride=(1, 1),
        #                                        padding=(0, 0),
        #                                        bias=True ),
                                    
        #                             nn.Tanh() 
        #                           )
        # l = (l - filter_size + 1)

        # self.conv6 = nn.Sequential( nn.Conv1d( in_channels=10 * channel_in,
        #                                        out_channels=5 * channel_in,
        #                                         kernel_size=(filter_size, 1),
        #                                         stride=(1, 1),
        #                                         padding=(0, 0),
        #                                         bias=True ), 
                                    
        #                             nn.Tanh() )
        
        # l = (l - filter_size + 1)
        self.fc = nn.Sequential( nn.Linear( l * 3 * channel_in, 100, bias=True) ,
                                 nn.Dropout(dropout),
                                 nn.Linear( 100, 100, bias=True) ,
                                 nn.Dropout(dropout),                          
                                 
                                 nn.Linear( 100, length_label * channel_out, bias=True ) 
                                )
        
    def forward(self, inputs):
        """
        Forward pass of the Neural Network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - inputs: PyTorch input Variable
        """
        
        preds = None
   
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        # out = self.conv4(out)
        # out = self.conv5(out)
        # out = self.conv6(out)
        
        batch_size, channels, height, width = out.shape 
        out = out.view(-1, channels * height * width)
        preds = self.fc(out)
      
        return preds.float()


