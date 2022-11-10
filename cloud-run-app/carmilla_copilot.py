from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class CarmillaCopilot():
    vocab = [
    '\n', ' ', '!', '"', "'", ',', '-', '.', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    def __init__(self, log):
        self._log = log
        if tf.config.experimental.list_physical_devices('GPU'):
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                self.log.info(f"{len(gpus)}, Physical GPUs, {len(logical_gpus)}, Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        self.model = self._build_model(
              vocab_size = len(CarmillaCopilot.vocab),
            )
        self.model.load_weights("models/ckpt_200")

    def __call__(self, start_string):
        return self._generate_text(self.model, start_string)
    
    def _build_model(self, vocab_size, embedding_dim=256, rnn_units=1024, batch_size=1):
        self._log.info("building tf model")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                vocab_size, embedding_dim,
                batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(
                rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(
                vocab_size, 
                name="t_out")
          ])
        return model

    def _generate_text(self, _model, start_string, temperature = 0.001):
        self._log.info(f"generating text from: {start_string}")
        # Number of characters to generate
        num_generate = 1000

        # Converting our start string to numbers (vectorizing)
        input_eval = [self.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        # Empty string to store our results
        text_generated = []
        period_count = 0
        _model.reset_states()
        for i in range(num_generate):
            predictions = _model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
            
            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)
            
            text_generated.append(self.idx2char[predicted_id])

            # We're stopping early here if there are several periods just to speed up our prediction
            if predicted_id == self.char2idx["."]:
                period_count += 1
            if period_count >= 5:
                break

        return (start_string + ''.join(text_generated))
