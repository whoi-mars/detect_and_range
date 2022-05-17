import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CNNRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lstm_dropout=None, fc_dropout=None, pretrain=False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc_dropout = fc_dropout
        
        # Get convolutional parts of resnet18
        resnet_full = models.resnet18(pretrained=pretrain)
        self.cnn = nn.Sequential(*list(resnet_full.children())[:-2])
        
        # Run random example through to get output shape
        sample = torch.randn((1, 3, input_size[0], input_size[1]))
        _, self.cnn_out_features, self.cnn_out_h, self.cnn_out_w = self.cnn(sample).shape
        
        self.lstm = nn.LSTM(input_size=3584, hidden_size=hidden_size, num_layers=num_layers, dropout=lstm_dropout, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)
        
        if fc_dropout is not None:
            self.dropout = nn.Dropout(p=fc_dropout)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        # Get spatiotemporal features where "space" is frequency + magnitude
        out = self.cnn(x)
        
        # Make the out (batch, time_index, freq_features*num_feature_maps)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(out.size(0), self.cnn_out_h, self.cnn_out_w*self.cnn_out_features)
        
        # Run through RNN and take output from last hidden/cell state
        h0 = torch.zeros(self.num_layers, out.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, out.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(out, (h0,c0))
        out = out[:,-1,:]
        
        # dropout
        if self.fc_dropout is not None:
            out = self.dropout(out)
        
        # Run through linear layer
        out = self.fc(out)
        
        # Run classification outputs through sigmoid
        out[:,1] = self.sigmoid(out[:,1])
        
        return out

class CNNRNN_multi_fc(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lstm_dropout=None, fc_dropout=None, pretrain=False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc_dropout = fc_dropout
        
        # Get convolutional parts of resnet18
        resnet_full = models.resnet18(pretrained=pretrain)
        self.cnn = nn.Sequential(*list(resnet_full.children())[:-2])
        
        # Run random example through to get output shape
        sample = torch.randn((1, 3, input_size[0], input_size[1]))
        _, self.cnn_out_features, self.cnn_out_h, self.cnn_out_w = self.cnn(sample).shape
        
        self.lstm = nn.LSTM(input_size=3584, hidden_size=hidden_size, num_layers=num_layers, dropout=lstm_dropout, batch_first=True)
        self.fc11 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc12 = nn.Linear(in_features=hidden_size, out_features=1)
        self.fc21 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc22 = nn.Linear(in_features=hidden_size, out_features=1)
        
        if fc_dropout is not None:
            self.dropout1 = nn.Dropout(p=fc_dropout)
            self.dropout2 = nn.Dropout(p=fc_dropout)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        
        # Get spatiotemporal features where "space" is frequency + magnitude
        out = self.cnn(x)
        
        # Make the out (batch, time_index, freq_features*num_feature_maps)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(out.size(0), self.cnn_out_h, self.cnn_out_w*self.cnn_out_features)
        
        # Run through RNN and take output from last hidden/cell state
        h0 = torch.zeros(self.num_layers, out.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, out.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(out, (h0,c0))
        out = out[:,-1,:]
        
        # Run through linear layer
        out11 = self.fc11(out)
        out11 = self.relu(out11)
        if self.fc_dropout is not None:
            out11 = self.dropout1(out11)
        out12 = self.fc12(out11)
        
        out21 = self.fc21(out)
        out21 = self.relu(out21)
        if self.fc_dropout is not None:
            out21 = self.dropout2(out21)
        out22 = self.sigmoid(self.fc22(out21))
        
        # concatinate fc outputs
        out_final = torch.cat((out12, out22),1)
        
        return out_final

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # Entry flow layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv5_skip = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')

        # Middle flow layers
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.conv10 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')

        # Out flow layers
        self.conv11 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.fc1 = nn.Linear(in_features=256, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=2)
        self.d = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Entry flow
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x_skip = F.relu(self.conv5_skip(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x_entry_out = x + x_skip

        # Middle flow
        x = F.relu(self.conv8(x_entry_out))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x_mid_out = x + x_entry_out

        # Out flow
        x = F.relu(self.conv11(x_mid_out))
        x = F.relu(self.conv12(x))
        x = x.mean(dim=(2,3))
        x = F.relu(self.fc1(x))
        x = self.d(x)
        x = self.fc2(x)
        x[:,1] = self.sigmoid(x[:,1])

        return x

class BNet(nn.Module):

    def __init__(self):
        super().__init__()
        # Entry flow layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, padding='same')
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')

        # Middle flow layers
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')

        self.conv14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')

        self.conv17 = nn.Conv2d(in_channels=512, out_channels=640, kernel_size=3)
        self.conv18 = nn.Conv2d(in_channels=640, out_channels=640, kernel_size=3)

        # Out flow layers
        self.conv19 = nn.Conv2d(in_channels=640, out_channels=768, kernel_size=3)
        self.conv20 = nn.Conv2d(in_channels=768, out_channels=1024, kernel_size=3)
        self.d1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=1024, out_features=1)

        self.conv21 = nn.Conv2d(in_channels=640, out_channels=768, kernel_size=3)
        self.conv22 = nn.Conv2d(in_channels=768, out_channels=1024, kernel_size=3)
        self.d2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=1024, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Entry flow
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x_skip = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x_entry_out = x + x_skip

        # Middle flow
        x = F.relu(self.conv8(x_entry_out))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x_mid_skip_1 = x + x_entry_out

        x = F.relu(self.conv11(x_mid_skip_1))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x_mid_skip_2 = x + x_mid_skip_1

        x = F.relu(self.conv14(x_mid_skip_2))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = x + x_mid_skip_2
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = F.relu(self.conv17(x))
        x_mid_out = F.relu(self.conv18(x))

        # Out flow
        # range
        x = F.relu(self.conv19(x_mid_out))
        x = F.relu(self.conv20(x))
        x = x.mean(dim=(2,3))
        x = self.d1(x)
        x_range = self.fc1(x)

        # class
        x = F.relu(self.conv21(x_mid_out))
        x = F.relu(self.conv22(x))
        x = x.mean(dim=(2,3))
        x = self.d2(x)
        x_class = self.sigmoid(self.fc2(x))

        out = torch.cat((x_range, x_class), 1)

        return out