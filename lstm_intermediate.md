# Long Short-Term Memory (LSTM) Networks

## Why LSTMs exist
Recurrent neural networks process sequences, but training over long sequences can suffer from vanishing gradients. LSTMs introduce a memory cell and gates to help preserve useful information over time.

## Forget gate
The forget gate decides how much of the previous cell state to retain. It uses a sigmoid activation to output values between zero and one. A value near zero removes information, while a value near one preserves it.

## Input gate
The input gate determines how much new candidate information should enter the cell state. This lets the network update memory selectively rather than replacing it at every time step.

## Output gate
The output gate controls which parts of the cell state influence the hidden state passed to the next time step and to downstream layers.

## Practical applications
LSTMs are useful for time-series prediction, sequence classification, and language modeling, though transformers are now common for many language tasks.
