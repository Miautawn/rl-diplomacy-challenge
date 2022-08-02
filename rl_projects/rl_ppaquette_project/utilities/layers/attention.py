import numpy as np
import torch
import torch.nn as nn

class StaticAttentionWrapper(nn.Module):
    """ Wraps another `RNNCell` with attention. """
    
    def __init__(self, cell, attention_memory_size, attention_memory_time, batch_size, probability_function=None, score_mask_value=None,
                cell_input_function=None, output_attention=False):
        """ Constructs an AttentionWrapper with static alignments (attention weights)

            :param cell: An instance of `RNNCell`.
            :param attention_memory_size: 
            :param attention_memory_time: 
            :param batch_sie: the size of the batch
            :param probability_function: A `callable`.  Converts the score to probabilities.  The default is @{tf.nn.softmax}.
            :param score_mask_value:  The mask value for score before passing into `probability_fn`. Default is -inf.
            :param cell_input_function: (optional) A `callable` to aggregate attention.
                                  Default: `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
            :param output_attention: If true, outputs the attention, if False outputs the cell output.
        """
        super(StaticAttentionWrapper, self).__init__()
        
        self.cell = cell
        self.attention_memory_size = attention_memory_size
        self.attention_memory_time = attention_memory_time
        self.batch_size = batch_size
        self.probability_function = probability_function
        self.score_mask_value = score_mask_value
        self.output_attention = output_attention
            
        # Validating probability Function
        if self.probability_function is None:
            self.probability_function = nn.Softmax
            
        # TODO: validate output function or something
            
        # Validating score_mask_value
        if self.score_mask_value is None:
            self.score_mask_value = -np.inf
            
        # TODO: validate cell function or something
        
        #Storing the initial values placeholders
        self.initial_alignment = None
        self.initial_attention = None
        
        # Storing zero placeholders
        self.zero_cell_output = torch.zeros([self.batch_size, self.cell.hidden_size])
        self.zero_attention = torch.zeros([self.batch_size, self.attention_memory_size])
        # self.zero_state = self.zero_state(batch_size, dtypes.float32) # TODO: make this inheritable
        self.zero_alignment = torch.zeros([self.batch_size, self.attention_memory_time])
        
    def __getattr__(self, name):
        """
        As this is a cell wrapper, direct unknown member accesses to the cell 
        """
        try:
            # this is needed to avoid infinite recursion
            return super(StaticAttentionWrapper, self).__getattr__(name)
        except AttributeError:
            return getattr(self.cell, name)
        
        
    def _compute_attention(self, alignments, memory):
        """Computes the attention and alignments for a given attention_mechanism."""
        # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
        expanded_alignments = alignments[:, None, :].copy()

        # Context (Attention in this case) is the inner product of alignments and values along the
        # memory time dimension.
        # alignments shape is  [batch_size, 1, memory_time]
        # memory is [batch_size, memory_time, memory_size]
        # the batched matmul is over memory_time, so the output shape is [batch_size, 1, memory_size].
        # we then squeeze out the singleton dim.
        context = torch.matmul(expanded_alignments, memory)
        context = torch.squeeze(context, dim=1)
        
        # NOTE: you could potentially pass the context through fully-connected layer
        # in our case, we will just use the context as the attention
        attention = context
        
        return attention, alignments
    
    
    def get_zero_state(self):
        """ Return an initial (zero) state dict for this `StaticAttentionWrapper`.
            :return: A dictionary containing zeroed out tensors of this and dowstream objects
        """
        
        return {
            "cell_state": self.cell.get_zero_state(),
            "attention": self
        }
        return AttentionWrapperState(cell_state=self._cell.zero_state(batch_size, dtype),
                                     time=array_ops.zeros([], dtype=dtypes.int32),
                                     attention=self._initial_attention,
                                     alignments=self._initial_alignment,
                                     attention_state=self._initial_alignment,
                                     alignment_history=())
        
        
    def forward(memory, alignments, memory_sequence_lengths):
        """ Perform a step of attention-wrapped RNN
            :param memory: The memory to query [batch_size, memory_time, memory_size]
            :param alignments: A tensor of probabilities of shape [batch_size, time_steps, memory_time]
            :param sequence_length: Sequence lengths for the batch entries in memory. Size (b,)
            
            :param inputs: (Possibly nested tuple of) Tensor, the input at this time step.
            :param state: An instance of `AttentionWrapperState` containing tensors from the previous time step.
            :return: A tuple `(attention_or_cell_output, next_state)`, where:
                    - `attention_or_cell_output` depending on `output_attention`.
                    - `next_state` is an instance of `AttentionWrapperState` containing the state calculated at
                       this time step.
        """
                
        # Validating sequence_length values
        if any(memory_sequence_lengths <= 0):
            raise ValueError("All values in memory_sequence_lengths must greater than zero.")
            
        # Storing initial values
        alignments = torch.transpose(alignments, (0,1)) # [batch, max_time, memory_time] -> [max_time, batch, memory_time]
        self.initial_alignment = alignments[0]
        self.initial_attention = self._compute_attention(self.initial_alignment, memory)[0]

#         next_time = state.time + 1
#         finished = (next_time >= self._sequence_length)
#         all_finished = math_ops.reduce_all(finished)

#         def get_next_alignments():
#             """ Returns the next alignments """
#             next_align = self._alignments_ta.read(next_time)
#             with ops.control_dependencies([next_align]):
#                 return array_ops.identity(next_align)

#         # Calculate the true inputs to the cell based on the previous attention value.
#         cell_inputs = self._cell_input_fn(inputs, state.attention)
#         cell_state = state.cell_state
#         cell_output, cell_state = self._cell(cell_inputs, cell_state)

#         # Computing context
#         next_alignments = control_flow_ops.cond(all_finished,
#                                                 true_fn=lambda: self._zero_alignment,
#                                                 false_fn=get_next_alignments)
#         attention, _ = self._compute_attention(next_alignments, self._memory)

#         next_state = AttentionWrapperState(time=next_time,
#                                            cell_state=cell_state,
#                                            attention=attention,
#                                            alignments=next_alignments,
#                                            attention_state=next_alignments,
#                                            alignment_history=())

#         if self._output_attention:
#             return attention, next_state
#         return cell_output, next_state
        
        
    
    