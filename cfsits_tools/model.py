import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LSTMFCN(nn.Module):
    # PyTorch translation of the Keras code in https://github.com/sktime and https://github.com/houshd/MLSTM-FCN
    def __init__(self, nb_classes, dim, dropout=0.8, kernel_sizes=(8,5,3),
                 filter_sizes=(128, 256, 128), lstm_size=8, attention=False):
        super(LSTMFCN, self).__init__()

        # self.attention = attention

        self.LSTM = nn.LSTM(dim, lstm_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        conv_layers = []
        for i in range(len(filter_sizes)):
            conv_layers.append(nn.LazyConv1d(filter_sizes[0], kernel_sizes[0], padding="same")) # keras: kernel_initializer="he_uniform"
            conv_layers.append(nn.BatchNorm1d(filter_sizes[0]))
            conv_layers.append(nn.ReLU())
            if i < len(filter_sizes):
                conv_layers.append(SqueezeExciteBlock(filter_sizes[i]))

        self.conv_layers = nn.Sequential(*conv_layers)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(nb_classes),
            # nn.Softmax(dim=1) # already performed inside CrossEntropyLoss
        )

    def forward(self, input):
        # Dimension shuffle: input.permute(0,2,1)
        # Unecessary, since LSTM already takes (batch, seq, feature) reversed wrt to our input (batch, var, time), and also wrt the conv1d convention. 
        # We want to give all timesteps to LSTM at each step (as proposed in the paper).
        whole_seq_output, _ = self.LSTM(input) 
        x = whole_seq_output[:,-1,:] # Take only last time step of size (batch, lstm_size), as pytorch returns the whole sequence
        x = self.dropout(x)

        y = self.conv_layers(input)
        y = self.gap(y)

        output = torch.cat((x,torch.squeeze(y)),dim=1)
        return self.fc(output)

class SqueezeExciteBlock(nn.Module):
    def __init__(self, input_channels):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // 16)
        self.fc2 = nn.Linear(input_channels // 16, input_channels)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = x_se.view(x_se.size(0), -1)
        x_se = F.relu(self.fc1(x_se))
        x_se = torch.sigmoid(self.fc2(x_se))
        x_se = x_se.view(x_se.size(0), -1, 1)
        x = x * x_se
        return x


class InceptionLayer(nn.Module):
    # PyTorch translation of the Keras code in https://github.com/hfawaz/dl-4-tsc
    def __init__(self, nb_filters=32, use_bottleneck=True,
                 bottleneck_size=32, kernel_size=40):
        super(InceptionLayer, self).__init__()

        # self.in_channels = in_channels
        kernel_size_s = [(kernel_size) // (2 ** i) for i in range(3)] # = [40, 20, 10]
        kernel_size_s = [x+1 for x in kernel_size_s] # Avoids warning about even kernel_size with padding="same"
        self.bottleneck_size = bottleneck_size
        self.use_bottleneck = use_bottleneck


        # Bottleneck layer
        self.bottleneck = nn.LazyConv1d(self.bottleneck_size, kernel_size=1,
                                    stride=1, padding="same", bias=False)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.bottleneck_conv = nn.LazyConv1d(nb_filters, kernel_size=1,
                                         stride=1, padding="same", bias=False)

        # Convolutional layer (several filter lenghts)
        self.conv_list = nn.ModuleList([])
        for i in range(len(kernel_size_s)):
            # Input size could be self.in_channels or self.bottleneck_size (if bottleneck was applied)
            self.conv_list.append(nn.LazyConv1d(nb_filters, kernel_size=kernel_size_s[i],
                                            stride=1, padding='same', bias=False))

        self.bn = nn.BatchNorm1d(4*self.bottleneck_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        in_channels = input.shape[-2]
        if self.use_bottleneck and int(in_channels) > self.bottleneck_size:
            input_inception = self.bottleneck(input)
        else:
            input_inception = input

        max_pool = self.max_pool(input)
        output = self.bottleneck_conv(max_pool)
        for conv in self.conv_list:
            output = torch.cat((output,conv(input_inception)),dim=1)

        output = self.bn(output)
        output = self.relu(output)

        return output


class InceptionBranch(nn.Module):
    # PyTorch translation of the Keras code in https://github.com/hfawaz/dl-4-tsc
    def __init__(self, nb_filters=32, use_residual=True,
                 use_bottleneck=True, bottleneck_size=32, depth=6, kernel_size=41):
        super(InceptionBranch, self).__init__()

        self.use_residual = use_residual

        # Inception layers
        self.inception_list = nn.ModuleList(
            [InceptionLayer(nb_filters,use_bottleneck, bottleneck_size, kernel_size) for _ in range(depth)])
        # Explicit input sizes (i.e. without using Lazy layers). Requires n_var passed as a constructor input
        # self.inception_list = nn.ModuleList([InceptionLayer(n_var, nb_filters,use_bottleneck, bottleneck_size, kernel_size) for _ in range(depth)])
        # for _ in range(1,depth):
        #     inception = InceptionLayer(4*nb_filters,nb_filters,use_bottleneck, bottleneck_size, kernel_size)
        #     self.inception_list.append(inception)

        # Fully-connected layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Flatten()

        # Shortcut layers
        # First residual layer has n_var channels as inputs while the remaining have 4*nb_filters
        self.conv = nn.ModuleList([
            nn.LazyConv1d(4*nb_filters, kernel_size=1,
                            stride=1, padding="same", bias=False)
            for _ in range(int(depth/3))
        ])
        self.bn = nn.ModuleList([nn.BatchNorm1d(4*nb_filters) for _ in range(int(depth/3))])
        self.relu = nn.ModuleList([nn.ReLU() for _ in range(int(depth/3))])

    def _shortcut_layer(self, input_tensor, out_tensor, id):
        shortcut_y = self.conv[id](input_tensor)
        shortcut_y = self.bn[id](shortcut_y)
        x = torch.add(shortcut_y, out_tensor)
        x = self.relu[id](x)
        return x

    def forward(self, x):
        input_res = x

        for d, inception in enumerate(self.inception_list):
            x = inception(x)

            # Residual layer
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res,x, int(d/3))
                input_res = x

        gap_layer = self.gap(x)
        return self.out(gap_layer)


class Inception(nn.Module):
    # PyTorch translation of the Keras code in https://github.com/hfawaz/dl-4-tsc
    def __init__(self, nb_classes, nb_filters=32, use_residual=True,
                 use_bottleneck=True, bottleneck_size=32, depth=6, kernel_size=41):
        super(Inception, self).__init__()

        self.use_residual = use_residual

        # Inception layers
        self.inception_list = nn.ModuleList(
            [InceptionLayer(nb_filters,use_bottleneck, bottleneck_size, kernel_size) for _ in range(depth)])
        # Explicit input sizes (i.e. without using Lazy layers). Requires n_var passed as a constructor input
        # self.inception_list = nn.ModuleList([InceptionLayer(n_var, nb_filters,use_bottleneck, bottleneck_size, kernel_size) for _ in range(depth)])
        # for _ in range(1,depth):
        #     inception = InceptionLayer(4*nb_filters,nb_filters,use_bottleneck, bottleneck_size, kernel_size)
        #     self.inception_list.append(inception)

        # Fully-connected layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(nb_classes),
            # nn.Softmax(dim=1) # already performed inside CrossEntropyLoss
        )

        # Shortcut layers
        # First residual layer has n_var channels as inputs while the remaining have 4*nb_filters
        self.conv = nn.ModuleList([
            nn.LazyConv1d(4*nb_filters, kernel_size=1,
                            stride=1, padding="same", bias=False)
            for _ in range(int(depth/3))
        ])
        self.bn = nn.ModuleList([nn.BatchNorm1d(4*nb_filters) for _ in range(int(depth/3))])
        self.relu = nn.ModuleList([nn.ReLU() for _ in range(int(depth/3))])

    def _shortcut_layer(self, input_tensor, out_tensor, id):
        shortcut_y = self.conv[id](input_tensor)
        shortcut_y = self.bn[id](shortcut_y)
        x = torch.add(shortcut_y, out_tensor)
        x = self.relu[id](x)
        return x

    def forward(self, x):
        input_res = x

        for d, inception in enumerate(self.inception_list):
            x = inception(x)

            # Residual layer
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res,x, int(d/3))
                input_res = x

        gap_layer = self.gap(x)
        return self.fc(gap_layer)


class S2Branch(nn.Module):
    def __init__(self, dropout_rate = 0.0, n_channels=64, hidden_activation='relu',
                 output_activation='softmax',
                 name='S2Branch',
                 **kwargs):
        super(S2Branch, self).__init__(**kwargs)

        kernel_size = 5
        padding = 1
        self.conv1 = nn.LazyConv1d(64,kernel_size,padding=padding)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.LazyConv1d(64,kernel_size,padding=padding)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dp2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.LazyConv1d(64,kernel_size,padding=padding)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dp3 = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()

        self.gap = nn.AdaptiveAvgPool1d(1)

        # self.convblock1 = ConvBlock(n_channels,kernel_size,padding,dropout_rate)
        # self.convblock2 = ConvBlock(n_channels,kernel_size,padding,dropout_rate)
        # self.convblock3 = ConvBlock(n_channels,kernel_size,padding,dropout_rate)

        # self.flatten = nn.Flatten()
        self.gap = nn.AdaptiveAvgPool1d(1)
        
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

        # output1 = self.convblock1(inputs)
        # output2 = self.convblock2(output1)
        # output3 = self.convblock2(output2)

        output = self.gap(output3)
        #return self.flatten(output3)
        return torch.squeeze(output, dim=-1)


# class ConvBlock(nn.Module):
#     def __init__(self, out_channels=64, kernel_size=5, padding=1, dropout_rate=0):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.LazyConv1d(out_channels,kernel_size,padding=padding)
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU()
#         self.dp = nn.Dropout(dropout_rate)

#     def forward(self, inputs):
#         output1 = self.conv(inputs)
#         output1 = self.bn(output1)
#         output1 = self.relu(output1)
#         return self.dp(output1)


class BinaryClassif(nn.Module):
    def __init__(self, dropout_rate = 0.0, hidden_activation='relu', output_activation='softmax',
                 name='BinaryClassif',**kwargs):
        super(BinaryClassif, self).__init__(**kwargs)
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
    def __init__(self,
                 dropout_rate=0.0,
                 encoder='Inception',
                 input_dim = None,
                 encoder_params=None,
                 hidden_activation='relu',
                 output_activation='softmax',
                 name='Discr',
                 **kwargs):
        super(Discr, self).__init__(**kwargs)
        if encoder == 'TempCNN':
            encoder_class = S2Branch
            if encoder_params is None:
                n_channels = 64 if input_dim is None else int(64*input_dim/24)
                encoder_params = dict(dropout_rate=dropout_rate, n_channels=n_channels)
        elif encoder == 'Inception':
            encoder_class = InceptionBranch
            encoder_params = encoder_params or dict()
        elif encoder == 'MLP':
            encoder_class = MLPBranch
            if encoder_params is None:
                n_units = 128 if input_dim is None else int(128*input_dim/24)
                encoder_params = dict(dropout_rate=dropout_rate, n_units=n_units)
        self.encoder = encoder_class(**encoder_params)


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
    def __init__(self,
                 out_dim, 
                 input_dim = None,
                 dropout_rate = 0.0, n_var=1, 
                 hidden_activation='relu', output_activation=None,
                 name='Noiser', shrink=False, base_arch='MLP',
                 **kwargs):
        super(Noiser, self).__init__(**kwargs)
        self.shrink = shrink
        hidden_dim = 128*n_var if input_dim is None else int(128*n_var*out_dim/24)
        # for now only MLP arch is supported
        self.body=MLPBranch(
            dropout_rate=dropout_rate, 
            n_units=hidden_dim,
            hidden_activation=hidden_activation,
            batch_norm=True)
        
        self.dp2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.LazyLinear(out_dim)
        self.tanh = nn.Tanh()

        if self.shrink:
            self.ST = nn.Softshrink(0.1)

        self.unflatten = nn.Unflatten(-1,(n_var,int(out_dim/n_var)))

    def forward(self, inputs):
        output = self.body(inputs)

        output = self.dp2(output)
        
        output = self.dense3(output)
        #if self.output_activation is not None:
        output = self.tanh(output)
        if self.shrink:
            output = self.ST(output)
        return self.unflatten(output)


class MLPBranch(nn.Module):
    def __init__(self, 
                 dropout_rate = 0.0, 
                 n_units=128,
                 hidden_activation='relu',
                 batch_norm=False,
                 name='MLP',
                 **kwargs):
        super(MLPBranch, self).__init__(**kwargs)
        self.dense1 = nn.LazyLinear(n_units)
        if hidden_activation == 'relu':
            self.activ1 = nn.ReLU()
            self.activ2 = nn.ReLU()
        elif hidden_activation == 'tanh':
            self.activ1 = nn.Tanh()
            self.activ2 = nn.Tanh()
        self.dp1 = nn.Dropout(dropout_rate)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(n_units)
            self.bn2 = nn.BatchNorm1d(n_units)
        self.dense2 = nn.LazyLinear(n_units)
        self.flatten = nn.Flatten()


    def forward(self, inputs):
        output = self.flatten(inputs)
        output = self.dense1(output)
        if self.batch_norm:
            output = self.bn1(output)
        output = self.activ1(output)
        output = self.dp1(output)
        output = self.dense2(output)
        if self.batch_norm:
            output = self.bn2(output)
        output = self.activ2(output)
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

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.dp2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = self.dp3(output)

        output = self.flatten(output)
        #output = F.adaptive_avg_pool2d(output, (1, 1))
        #output = F.adaptive_max_pool2d(output, (1, 1))
        #output = output.squeeze()
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