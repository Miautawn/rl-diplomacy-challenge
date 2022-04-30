import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_

from settings import DATA_FEATURES
from utilities.state_space_functions import get_adjacency_matrix

class GraphConvolution(nn.Module):
    """ Performs a graph convolution (ArXiV 1609.02907) """

    def __init__(self, n_features, input_size, output_size, norm_adjacency, activation_function=None, residual=False, bias=False):
        super(GraphConvolution, self).__init__()
        
        self.activation_function = activation_function if activation_function is not None else lambda x: x
        self.norm_adjacency = norm_adjacency
        self.bias = bias
        self.var_w, self.var_b = None, None
        self.residual = residual

        # Initializing variables
        self.var_w = torch.empty((n_features, input_size, output_size))
        kaiming_normal_(self.var_w)
        
        if self.bias:
            self.var_b = torch.zeros((output_size))

    def forward(self, inputs):
        """ Actually performs the graph convolution """
        
        pre_act = torch.permute(inputs, (1, 0, 2))                  # (b, N, in )               => (N, b, in )
        pre_act = torch.matmul(pre_act, self.var_w)                 # (N, b, in) * (N, in, out) => (N, b, out)
        pre_act = torch.permute(pre_act, [1, 0, 2])                 # (N, b, out)               => (b, N, out)
        pre_act = torch.matmul(self.norm_adjacency, pre_act)        # (b, N, N) * (b, N, out)   => (b, N, out)
        

        # Adds the bias
        if self.bias:
            pre_act += self.var_b                                   # (b, N, out) + (1,1,out) => (b, N, out)

        # Applying activation fn and residual connection
        post_act = self.activation_function(pre_act)
        if self.residual:
            post_act += inputs
            
        return post_act                                             # (b, N, out)
    

class FilmGcnResBlock(nn.Module):
    """
    Following the design here https://arxiv.org/pdf/1709.07871.pdf
    """
    def __init__(self, n_features, input_size, output_size, norm_adjacency, activation_function = nn.ReLU(), residual = True):
        super(FilmGcnResBlock, self).__init__()
        
        assert input_size == output_size or not residual, 'For residual blocks, the in and out dims must be equal'
        
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.residual = residual
        
        self.graph_convolution = GraphConvolution(
            n_features = n_features,
            input_size=self.input_size,
            output_size=self.output_size,
            norm_adjacency=norm_adjacency,
            activation_function=None,
            residual=self.residual,
            bias=True
        )
        
        # setting to the batch_norm defaults in tensorflow
        self.bath_normalisation = nn.BatchNorm1d(n_features, eps = 0.001, momentum = 0.001)
        
    def forward(self, inputs, gamma, beta):
        gcn_in_dim = inputs.shape[-1]
        assert self.input_size == gcn_in_dim, 'The given input dimensions do not match the originally specified dimensions'

        gcn_result = self.graph_convolution(inputs)
        gcn_bn_result = self.bath_normalisation(gcn_result)
        film_result = gamma * gcn_bn_result + beta

        # Applying activation function and residual connection
        if self.activation_function is not None:
            film_result = self.activation_function(film_result)
        if self.residual:
            film_result += inputs
            
        return film_result


class GraphConvolutionEncoder(nn.Module):
    """
    Uses GCN with FiLM to encode board state or previous order states.
    This class is just an abstraction for 'GraphConvolution' and 'FilmGcnResBlock' classes.
    """
    def __init__(self, n_features, input_size, output_gcn_size, final_output_gcn_size, n_graph_conv_layers, batch_size):
        super(GraphConvolutionEncoder, self).__init__()
        
        self.n_graph_conv_layers = n_graph_conv_layers
        
        #constructing base normalised adjacency matrix
        norm_adjacency = self._preprocess_adjacency(get_adjacency_matrix())
        
        # make copies of the matrix along the a new axis of size of the batch
        self.norm_adjacency = torch.tile(norm_adjacency, (batch_size, 1, 1))
        
        # setting the layers
        self.linear = nn.Linear(in_features = input_size, out_features = output_gcn_size)
        xavier_uniform_(self.linear.weight)
        zeros_(self.linear.bias)
        
        self.relu = nn.ReLU()
        
        self.intermediate_film_gcn_batch_norm = FilmGcnResBlock(
            n_features = n_features,
            input_size = output_gcn_size,
            output_size = output_gcn_size,
            norm_adjacency = self.norm_adjacency,
            residual = True
        )
        
        self.last_film_gcn_batch_norm = FilmGcnResBlock(
            n_features = n_features,
            input_size = output_gcn_size,
            output_size = final_output_gcn_size,
            norm_adjacency = self.norm_adjacency,
            residual = False
        )
        
                
    def _preprocess_adjacency(self, adjacency_matrix):
        """
        Symmetrically normalize the adjacency matrix for graph convolutions.
        """
        # Computing A^~ = A + I_N
        adj_tilde = adjacency_matrix + np.eye(adjacency_matrix.shape[0])

        # Calculating the sum of each row
        sum_of_row = np.array(adj_tilde.sum(1))

        # Calculating the D tilde matrix ^ (-1/2)
        d_inv_sqrt = np.power(sum_of_row, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)

        # Calculating the normalized adjacency matrix
        norm_adj = adj_tilde.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return torch.tensor(norm_adj, dtype=torch.float32)
    

    # check if gammas and betas are changing
    def forward(self, state, film_gammas, film_betas):
        
        # Adding noise to break symmetry
        state = state + torch.empty(state.shape).normal_(std=0.01)
        
        graph_conv = self.linear(state)
        graph_conv = self.relu(graph_conv)
        
        # First and intermediate layers
        for layer_idx in range(self.n_graph_conv_layers - 1):
            graph_conv = self.intermediate_film_gcn_batch_norm(
                inputs=graph_conv,                                    # (b, NB_NODES, gcn_size)
                gamma=film_gammas[layer_idx],
                beta=film_betas[layer_idx]
            )
            
        # Last layer
        graph_conv = self.last_film_gcn_batch_norm(
            inputs=graph_conv,                                        # (b, NB_NODES, final_size)
            gamma=film_gammas[-1],
            beta=film_betas[-1]
        )

        return graph_conv
