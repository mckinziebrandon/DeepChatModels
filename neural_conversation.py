
# Goals for this file:
# 1. Get ubuntu corpus in format of input sentence -> response of arbitary lengths.
# 2. Create the seq2seq model that accepts sequences of tokens,  one token at a time, and predicts output tokens one at a time.
#       --> Specifically, it is fed single tokens at a time until reaching some <EOS>.
#       --> It then feeds the hidden state output ('context vector') to a decoder LSTM of basically the same architecture.
#       --> We generate a sequence of tokens from the decoder until getting the <EOS> output token.

# Questions:
# Can I even use keras for this? I'm not convinced that the Seq2Seq github actually supports variable length inputs & outputs. Checking now for answer:
# Apparently this is done by :
# model = Seq2Seq(batch_input_shape=(16, 7, 5), hidden_dim=10, output_length=8, output_dim=20, depth=4)
# model.compile(loss='mse', optimizer='rmsprop')