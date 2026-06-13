# -*- coding: utf-8 -*-
"""
Edited from class example

Marvin
"""
import torch.nn as nn
import torch.nn.functional as F

class ClayClassifierModule(nn.Module):
    def __init__(self, 
        n_input_channels: int, 
        sequence_length: int, 
        n_output_probs: int, 
        conv_layer_sizes=(32, 64), 
        conv_kernel_sizes=(3, 3), 
        act_fn_maxpool=F.relu, 
        dense_layer_sizes=(100, 100), 
        act_fn_dense=F.relu, 
        dropout=0.5,
        apply_softmax=True
    ):
        
        super().__init__()
        self.sequence_length = sequence_length
        self.conv_layer_sizes = conv_layer_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.act_fn_maxpool = act_fn_maxpool
        self.dense_layer_sizes = dense_layer_sizes
        self.act_fn_dense = act_fn_dense
        self.apply_softmax = apply_softmax

        # Conv network initialization
        self.conv_network = nn.ModuleList()
        self.conv_network.append(
            nn.Conv1d(
                in_channels=n_input_channels, 
                out_channels=conv_layer_sizes[0],
                kernel_size=conv_kernel_sizes[0]
            )
        )
        self.conv_network.append(nn.MaxPool1d(2))

        # Rest of the Conv network
        for idx in range(len(self.conv_layer_sizes) - 1):
            self.conv_network.append(
                nn.Conv1d(
                    in_channels=self.conv_layer_sizes[idx],
                    out_channels=self.conv_layer_sizes[idx + 1],
                    kernel_size=self.conv_kernel_sizes[idx + 1]
                )
            )
            self.conv_network.append(nn.Dropout(p=dropout))
            self.conv_network.append(nn.MaxPool1d(2))

        # Dense network after Conv network
        self.dense_network = nn.ModuleList()
        self.dense_network.append(nn.Linear(self.calc_dense_n_inputs(), self.dense_layer_sizes[0]))

        # Rest of the Dense network
        for idx in range(len(self.dense_layer_sizes) - 1):
            self.dense_network.append(
                nn.Linear(
                    self.dense_layer_sizes[idx], 
                    self.dense_layer_sizes[idx + 1]
                )
            )

        self.output = nn.Linear(self.dense_layer_sizes[-1], n_output_probs)

    def calc_dense_n_inputs(self):
        final_size = self.sequence_length
        for kernel_size in self.conv_kernel_sizes:              # simplification of kernel size handling
            final_size = (final_size - (kernel_size - 1)) // 2  # combined conv and maxpoo, integer division
            
        return int(self.conv_layer_sizes[-1] * final_size)

    def forward(self, X, **kwargs):
        # Pass through convolutional layers
        for layer in self.conv_network:
            if isinstance(layer, nn.MaxPool1d):
                X = self.act_fn_maxpool(layer(X))
            else:
                X = layer(X)
                
        X = X.view(-1, X.size(1) * X.size(2))  # Flatten for dense layers

        # Pass through dense layers
        for layer in self.dense_network:
            X = self.act_fn_dense(layer(X))
        
        output = self.output(X) 
        return F.softmax(output, dim=1) if self.apply_softmax else output


# Instantiate the model, providing sequence_length
model = ClayClassifierModule(n_input_channels=1, sequence_length=2000, n_output_probs=4)

    
    
    
    
    
    
    
    
    
    
    
    