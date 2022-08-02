import torch
import torch.nn as nn

from utilities.utility_functions import seeded_random 

class SeededDropoutWrapper(nn.Module):
    """A wrapper for an RNN cell which adds seeded dropout to inputs and outputs of the cell."""
    def __init__(
        self, cell, n_cell_states = 1, variational_recurrent=False, dropout_state_filter_visitor=None):
        """        
        If `variational_recurrent` is set to `True` (**NOT** the default behavior), then the same dropout mask is
        applied at every step, as described in:

            Y. Gal, Z Ghahramani.    "A Theoretically Grounded Application of Dropout in
            Recurrent Neural Networks".    https://arxiv.org/abs/1512.05287

        Otherwise a different dropout mask is applied at every time step.
            
        Note, by default (unless a custom `dropout_state_filter` is provided), the memory state (`c` component
        of any `LSTMStateTuple`) passing through a `DropoutWrapper` is never modified.    This behavior is
        described in the above article.
            
        (Cited from the source code: https://github.com/diplomacy/research/blob/6f6b4bd74372acbee15f24a73dcf7780f1cae9de/diplomacy_research/models/layers/dropout.py)
        """
        super(SeededDropoutWrapper, self).__init__()
            
        self.variational_recurrent = variational_recurrent
        self.cell = cell
        self.n_cell_states = n_cell_states
        
        # creating the isolated pytorch random generator.
        # this will allow us to decouple this wrapper class and whole model random actions
        self.random_generator = torch.Generator()
        self.random_generator.manual_seed(133742)
        
    def __getattr__(self, name):
        """
        As this is a cell wrapper, direct unknown member accesses to the cell 
        """
        try:
            # this is needed to avoid infinite recursion
            return super(SeededDropoutWrapper, self).__getattr__(name)
        except AttributeError:
            return getattr(self.cell, name)
        
        
    def forward(self, seeds, input_keep_probs, output_keep_probs, state_keep_probs):
        
        # reshape the probability vectors into (batch_size, 1) shape
        input_keep_probs = torch.reshape(input_keep_probs, (-1, 1))
        output_keep_probs = torch.reshape(output_keep_probs, (-1, 1))
        state_keep_probs = torch.reshape(state_keep_probs, (-1, 1))
        
        # Detecting if we skip computing these probabilities
        # TODO: EDIT
        skip_input_keep_probs = 0
        skip_output_keep_probs = 0
        skip_state_keep_probs = 0
        
        
        # Computing deterministic recurrent noise
        if self.variational_recurrent:  
            if skip_input_keep_probs == False:
                recurrent_input_noise = seeded_random(seeds, [self.cell.input_size], self.random_generator)
                
            if skip_state_keep_probs == False:
                recurrent_state_noise = [ seeded_random(seeds, [self.cell.hidden_size], self.random_generator) for _ in range(self.n_cell_states) ]
                recurrent_state_noise = torch.stack(recurrent_state_noise)    # shape: [n_states, batch_size, hidden_size]
                
            if skip_output_keep_probs == False:
                recurrent_output_noise = seeded_random(seeds, [self.cell.hidden_size], self.random_generator)
        
        return
    
    
class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.generator = torch.Generator()
        self.generator.manual_seed(2147483647)
        
    def forward(self):
        return torch.rand(4, generator = self.generator)

# class SeededDropoutWrapper(rnn_cell_impl.DropoutWrapper):
#     """Operator adding seeded dropout to inputs and outputs of the given cell."""

#     def __init__(self, cell, seeds, input_keep_probs=1.0, output_keep_probs=1.0, state_keep_probs=1.0,
#                  variational_recurrent=False, input_size=None, dtype=None, seed=None,
#                  dropout_state_filter_visitor=None):
#         """ Create a cell with added input, state, and/or output seeded dropout.

#             If `variational_recurrent` is set to `True` (**NOT** the default behavior), then the same dropout mask is
#             applied at every step, as described in:

#                 Y. Gal, Z Ghahramani.    "A Theoretically Grounded Application of Dropout in
#                 Recurrent Neural Networks".    https://arxiv.org/abs/1512.05287

#             Otherwise a different dropout mask is applied at every time step.

#             Note, by default (unless a custom `dropout_state_filter` is provided), the memory state (`c` component
#             of any `LSTMStateTuple`) passing through a `DropoutWrapper` is never modified.    This behavior is
#             described in the above article.

#             :param cell: an RNNCell, a projection to output_size is added to it.
#             :param seeds: A tensor representing the seed for each item in the batch. (Size: (batch,))
#             :param input_keep_probs: float, scalar tensor, or batch vector (b,). Input keep probabilities.
#             :param output_keep_probs: float, scalar tensor, or batch vector (b,). Output keep probabilities.
#             :param state_keep_probs: float, scalar tensor, or batch vector (b,). State keep probabilities (excl 'c')
#             :param variational_recurrent:  If `True`, same dropout pattern is applied across all time steps per run call
#             :param input_size: (optional) (possibly nested tuple of) `TensorShape` objects containing the depth(s) of
#                                the input tensors expected to be passed in to the `DropoutWrapper`.
#                                Required and used **iff** `variational_recurrent = True` and `input_keep_prob < 1`.
#             :param dtype: (optional) The `dtype` of the input, state, and output tensors.
#             :param seed: (optional) integer, the default randomness seed to use if one of the seeds is 0.
#             :param dropout_state_filter_visitor: Optional. See DropoutWrapper for description.
#         """
#         # pylint: disable=too-many-arguments
#         SeededDropoutWrapper.offset += 11828683
#         super(SeededDropoutWrapper, self).__init__(cell=cell,
#                                                    input_keep_prob=1.,
#                                                    output_keep_prob=1.,
#                                                    state_keep_prob=1.,
#                                                    variational_recurrent=False,
#                                                    input_size=input_size,
#                                                    dtype=dtype,
#                                                    seed=seed,
#                                                    dropout_state_filter_visitor=dropout_state_filter_visitor)

#         def _convert_to_probs_tensor(keep_probs):
#             """ Converts a keep_probs tensor to its broadcastable shape """
#             probs_tensor = ops.convert_to_tensor(keep_probs)
#             probs_tensor = gen_math_ops.maximum(0., gen_math_ops.minimum(1., probs_tensor))
#             return gen_array_ops.reshape(probs_tensor, [-1, 1])

#         # Converting to tensor
#         self._input_keep_probs = _convert_to_probs_tensor(input_keep_probs)
#         self._output_keep_probs = _convert_to_probs_tensor(output_keep_probs)
#         self._state_keep_probs = _convert_to_probs_tensor(state_keep_probs)

#         # Detecting if we skip computing those probs
#         self._skip_input_keep_probs = isinstance(input_keep_probs, float) and input_keep_probs == 1.
#         self._skip_output_keep_probs = isinstance(output_keep_probs, float) and output_keep_probs == 1.
#         self._skip_state_keep_probs = isinstance(state_keep_probs, float) and state_keep_probs == 1.

#         # Generating variational recurrent
#         self._seeds = seeds
#         self._variational_recurrent = variational_recurrent

#         enum_map_up_to = rnn_cell_impl._enumerated_map_structure_up_to                                                  # pylint: disable=protected-access

#         def batch_noise(input_dim, inner_offset, inner_seed):
#             """ Generates noise for variational dropout """
#             if not isinstance(input_dim, int):              # Scalar tensor - We can ignore it safely
#                 return None
#             return seeded_random(seeds,
#                                  offset=SeededDropoutWrapper.offset + inner_offset,
#                                  shape=[input_dim],
#                                  dtype=dtype,
#                                  seed=inner_seed)

#         # Computing deterministic recurrent noise
#         if variational_recurrent:
#             if dtype is None:
#                 raise ValueError('When variational_recurrent=True, dtype must be provided')

#             input_map_fn = lambda index, inp_shape: batch_noise(inp_shape, 127602767, self._gen_seed('input', index))
#             state_map_fn = lambda index, inp_shape: batch_noise(inp_shape, 31248361, self._gen_seed('state', index))
#             output_map_fn = lambda index, inp_shape: batch_noise(inp_shape, 71709719, self._gen_seed('output', index))

#             if not self._skip_input_keep_probs:
#                 if input_size is None:
#                     raise ValueError("When variational_recurrent=True and input_keep_prob < 1.0 or "
#                                      "is unknown, input_size must be provided")
#                 self._recurrent_input_noise = enum_map_up_to(input_size, input_map_fn, input_size)
#             self._recurrent_state_noise = enum_map_up_to(cell.state_size, state_map_fn, cell.state_size)
#             self._recurrent_output_noise = enum_map_up_to(cell.output_size, output_map_fn, cell.output_size)
