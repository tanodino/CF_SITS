import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class S2Branch(nn.Module):
    def __init__(self, n_class, dropout_rate = 0.0, hidden_activation='relu', output_activation='softmax',
                 name='S2Branch',
                 **kwargs):
        super(S2Branch, self).__init__(**kwargs)
        self.conv1 = nn.LazyConv1d(64,5,padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.LazyConv1d(64,5,padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dp2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.LazyConv1d(64,5,padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dp3 = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        '''
        self.dense = nn.lazyLinear(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.dp4 = nn.Dropout(dropout_rate)
        
        self.classif = nn.lazyLinear(n_class)
        self.softmax = nn.Softmax(dim=1)
        '''
        

    def forward(self, inputs):
        output1 = self.conv1(inputs)
        output1 = self.bn1(output1)
        output1 = self.relu1(output1)
        output1 = self.dp1(output1)
        
        output2 = self.conv2(output1)
        output2 = self.bn2(output2)
        output2 = self.relu2(output2)
        output2 = self.dp2(output2)

        output3 = self.conv3(output2)
        output3 = self.bn3(output3)
        output3 = self.relu3(output3)
        output3 = self.dp3(output3)

        #print("output1 ",output1.shape)
        output = self.gap(output3)
        #output = self.flatten(output3)
        #print("output.shape ",output.shape)
        return torch.squeeze(output)
        #print("output.shape ",output.shape)
        #return output



class BinaryClassif(nn.Module):
    def __init__(self, dropout_rate = 0.0, hidden_activation='relu', output_activation='softmax',
                 name='Classif',**kwargs):
        super(Classif, self).__init__(**kwargs)
        self.fc1 = nn.LazyLinear(256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.cl = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        
        output = self.fc1(inputs)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.drop1(output)
        output = self.cl(output)
        return self.sigmoid(output)

class Discr(nn.Module):
    def __init__(self, n_class, dropout_rate = 0.0, hidden_activation='relu', output_activation='softmax',
                 name='S2Classif',
                 **kwargs):
        super(S2Classif, self).__init__(**kwargs)
        self.encoder = S2Branch(dropout_rate)
        self.HeadBCl = BinaryClassif(dropout_rate=dropout_rate)

    def forward(self, inputs):
        output = self.encoder(inputs)
        return self.HeadBCl(output)



'''
class Discr(nn.Module):
    def __init__(self, dim, dropout_rate = 0.0, name='Discr', **kwargs):
        super(Discr, self).__init__(**kwargs)
        self.dense1 = nn.LazyLinear(32)
        self.bn1 = nn.BatchNorm1d(32)
        #self.relu = nn.ReLU(inplace=False)
        self.relu = nn.ReLU()
        self.dp1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        output = torch.squeeze(inputs)
        output = self.dense1(output)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.dp1(output)
        output = self.dense2(output)
        return self.sigmoid(output)
'''




class Noiser(nn.Module):
    def __init__(self, dim, dropout_rate = 0.0, hidden_activation='relu', output_activation=None,
                 name='Noiser',
                 **kwargs):
        super(Noiser, self).__init__(**kwargs)
        self.dense1 = nn.LazyLinear(128)
        #self.bn1 = nn.BatchNorm1d(128)
        self.tanh1 = nn.Tanh()#nn.ReLU()
        self.dp1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.LazyLinear(128)
        #self.bn2 = nn.BatchNorm1d(128)
        self.tanh2 = nn.Tanh()#nn.ReLU()
        self.dp2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.LazyLinear(dim)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        output = torch.squeeze(inputs)
        
        output = self.dense1(output)
        #output = self.bn1(output)
        output = self.tanh1(output)
        output = self.dp1(output)
        
        output = self.dense2(output)
        #output = self.bn2(output)
        output = self.tanh2(output)
        output = self.dp2(output)
        
        output = self.dense3(output)
        #if self.output_activation is not None:
        output = self.tanh(output)
        #return .1 * torch.unsqueeze(output,1)
        return torch.unsqueeze(output,1)


class MLPBranch(nn.Module):
    def __init__(self, n_class, dropout_rate = 0.0, hidden_activation='relu', output_activation='softmax',
                 name='MLP',
                 **kwargs):
        super(MLPBranch, self).__init__(**kwargs)
        self.dense1 = nn.LazyLinear(128)
        self.relu = nn.ReLU()
        self.dp1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.LazyLinear(128)
        self.flatten = nn.Flatten()


    def forward(self, inputs):
        output = self.flatten(inputs)
        output = self.dense1(output)
        output = self.relu(output)
        output = self.dp1(output)
        output = self.dense2(output)
        output = self.relu(output)
        return output


class CNN2DBranch(nn.Module):
    def __init__(self, dropout_rate = 0.0, hidden_activation='relu', output_activation='softmax',
                 name='CNN2DBranch',
                 **kwargs):
        super(CNN2DBranch, self).__init__(**kwargs)
        n_filters = 128
        n_filters2 = 128


        self.conv1 = torch.nn.LazyConv2d(n_filters, kernel_size=5, stride=1)
        #self.mp1 = nn.AvgPool2d((3,3), stride=2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.bn3 = nn.BatchNorm2d(n_filters)
        self.dp1 = nn.Dropout(dropout_rate)
        self.conv2 = torch.nn.LazyConv2d(n_filters, kernel_size=5, stride=1)
        self.dp2 = nn.Dropout(dropout_rate)
        self.conv3 = torch.nn.LazyConv2d(n_filters, kernel_size=5, stride=1)
        self.dp3 = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()


    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.dp1(output)
        #print(output.shape)
        #print(output.shape)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.dp2(output)
        #print(output.shape)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = self.dp3(output)
        #print(output.shape)

        output = self.flatten(output)
        #print(output.shape)
        #output = F.adaptive_avg_pool2d(output, (1, 1))
        #output = F.adaptive_max_pool2d(output, (1, 1))
        #output = output.squeeze()
        #print("GAP.shape ",output.shape)
        #exit()
        #print(output.shape)
        #exit()
        return output


class CNN2DClassif(nn.Module):
    def __init__(self, n_class, dropout_rate = 0.0, hidden_activation='relu', output_activation='softmax',
                 name='CNN2DClassif',
                 **kwargs):
        super(CNN2DClassif, self).__init__(**kwargs)
        self.encoder = CNN2DBranch(dropout_rate)
        self.fc1 = nn.LazyLinear(128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.LazyLinear(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.cl = nn.LazyLinear(n_class)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs):
        output = self.encoder(inputs)
        #output = self.cl(output)
        output = self.fc1(output)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.drop1(output)
        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.drop2(output)
        output = self.cl(output)
        #output = self.softmax(output)
        return output


class Regr(nn.Module):
    def __init__(self, dropout_rate = 0.0, hidden_activation='relu', output_activation='softmax',
                 name='Classif',**kwargs):
        super(Regr, self).__init__(**kwargs)
        self.fc1 = nn.LazyLinear(128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.LazyLinear(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.cl = nn.LazyLinear(1)

    def forward(self, inputs):
        output = self.fc1(inputs)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.drop1(output)

        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.drop2(output)

        output = self.cl(output)
        return output



class Classif(nn.Module):
    def __init__(self, n_class, dropout_rate = 0.0, hidden_activation='relu', output_activation='softmax',
                 name='Classif',**kwargs):
        super(Classif, self).__init__(**kwargs)
        self.fc1 = nn.LazyLinear(256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.cl = nn.LazyLinear(n_class)

    def forward(self, inputs):
        
        output = self.fc1(inputs)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.drop1(output)
        output = self.cl(output)
        return output


class MLPClassif(nn.Module):
    def __init__(self, n_class, dropout_rate = 0.0, hidden_activation='relu', output_activation='softmax',
                 name='MLPClassif',
                 **kwargs):
        super(MLPClassif, self).__init__(**kwargs)
        self.encoder = MLPBranch(dropout_rate)
        self.HeadCl = Classif(n_class=n_class, dropout_rate=dropout_rate)
    
    def forward(self, inputs):
        output = self.encoder(inputs)
        return self.HeadCl(output)

class S2Classif(nn.Module):
    def __init__(self, n_class, dropout_rate = 0.0, hidden_activation='relu', output_activation='softmax',
                 name='S2Classif',
                 **kwargs):
        super(S2Classif, self).__init__(**kwargs)
        self.encoder = S2Branch(dropout_rate)
        self.HeadCl = Classif(n_class=n_class, dropout_rate=dropout_rate)

    def forward(self, inputs):
        output = self.encoder(inputs)
        return self.HeadCl(output)