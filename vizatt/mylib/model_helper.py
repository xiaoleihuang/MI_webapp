import numpy as np
from scipy import stats
import os
import json

# keras
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import load_model
from keras import backend as K
import keras
from keras.layers import Layer
from keras.utils import CustomObjectScope
from keras.models import model_from_json

import sys

sys.setrecursionlimit(1000000)

try:
    import cPickle as pickle
except:
    import pickle

if keras.__version__.startswith('2'):
    import keras.initializers as initializations
else:
    import keras.initializations as initializations


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        self.init = initializations.get('glorot_uniform')

        self.W_regularizer = keras.regularizers.get(W_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.W_constraint = keras.constraints.get(W_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer=self.init,
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True
        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = K.dot(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def validate_inputs(input_mode, content, context, kmodel, kmodel_ctxt):
    """This function is to validate the inputs from the Django input form
    """
    if kmodel is None and kmodel_ctxt is None:
        return 'The models do not exist in Server! Plz contact Administrators!'

    if input_mode == None or input_mode not in ['1', '2']:
        return 'The input mode is unknown!'
    input_mode = int(input_mode)
    if content == None or len(content.strip()) < 2:
        return 'The input content can not be empty!'

    if input_mode == 2:
        if kmodel_ctxt is None:
            return 'The contexutal mode can not find the contextual model! Plz contact Administrators!'

        if context == None or len(context.strip()) < 2:
            return 'The contextual mode requires non-empty contextual input!'
    else:
        if kmodel is None:
            return 'The current mode can not find the content-only model! Plz contact Administrators!'
    return None


def load_model_config(config_path='./resources/settings.ini'):
    """Load configuration of the model, includes input, output, attention layers

        Args:
            config_path (str): the path of configuration file
        Return:
            model_config (dict): configurations of key-value pairs.
    """
    model_config = None
    if os.path.isfile(config_path):
        model_config = dict()
        try:
            with open(config_path) as config_file:
                for line in config_file:
                    line = line.strip()
                    if line.startswith('#'):
                        continue
                    if len(line) > 2:
                        infos = line.split('\t')
                        model_config[infos[0].strip()] = infos[1].strip()
        except:
            model_config = None
    if model_config is None:
        config_default = dict()
        config_default['keras_model'] = './resources/keras_model/rnn_attention_pure.model'
        config_default['keras_model_context'] = './resources/keras_model/rnn_attention_context.model'
        config_default['keras_tokenizer'] = './resources/keras_tokenizer.pkl'

        # for layer configurations
        config_default['content_input_layer'] = 'content_input'
        config_default['content_output_layer'] = 'content_bilstm'
        config_default['context_input_layer'] = 'context_input'
        config_default['context_output_layer'] = 'context_bilstm'

        return config_default
    return model_config


def load_keras_resources(model_path='./resources/rnn_attention_pure.model',
                         contextual_model_path='./resources/rnn_attention_context.json',
                         contextual_model_path_w='./resources/rnn_attention_context.h5',
                         tk_path='./resources/keras_tokenizer.pkl'):
    """It will loads resources of keras (contextual) model, keras tokenizer

    Args:
        model_path (str): keras model path without contxts
        contextual_model_path (str): contxtual keras model path
        contextual_model_path_w (str): weights of contxtual keras model path
        tk_path (str): path of tokenizer
    """
    kmodel = None
    kmodel_contxt = None
    ktk = None
    if os.path.isfile(model_path):
        with CustomObjectScope({'Attention': Attention}):
            kmodel = load_model(model_path)
    if os.path.isfile(contextual_model_path) and os.path.isfile(contextual_model_path_w):
        with CustomObjectScope({'Attention': Attention}):
            # load json of network architecture
            kmodel_contxt = model_from_json(
                json.dumps(json.load(open(contextual_model_path))),
                custom_objects={'Attention': Attention})
            # load its weights by using the same name
            kmodel_contxt.load_weights(contextual_model_path_w)
    if os.path.isfile(tk_path):
        ktk = pickle.load(open(tk_path, 'rb'))

    return kmodel, kmodel_contxt, ktk


def get_rnn_func(keras_model, config_att=None, input_mode=1):
    # if config is None, we will initialize with default; else it will use user-defined configuration.
    if config_att is None:
        config_att = dict()
        config_att['content_input_layer'] = 'content_input'
        # output from RNN,the layer is RNN
        config_att['content_output_layer'] = 'content_bilstm'
        # context configuration
        config_att['context_input_layer'] = 'context_input'
        config_att['context_output_layer'] = 'context_bilstm'

    rnn_layer_output_list = [
        K.function([keras_model.get_layer(config_att['content_input_layer']).input, K.learning_phase()],
                   [keras_model.get_layer(config_att['content_output_layer']).output])]

    if input_mode == 2:
        # for context
        rnn_layer_output_list.append(
            K.function([keras_model.get_layer(config_att['context_input_layer']).input, K.learning_phase()],
                       [keras_model.get_layer(config_att['context_output_layer']).output]))
    return rnn_layer_output_list


def get_att_output(rnn_layer_func, keras_model, single_input, layer_name='content_att'):
    if type(single_input) == list:
        single_input = np.asarray(single_input)
    sent_len = len([item for item in single_input[0] if item != 0])
    output = rnn_layer_func([single_input, 0])[0]  # test mode
    eij = np.dot(output, keras_model.get_layer(layer_name).get_weights()[0])
    eij += keras_model.get_layer(layer_name).get_weights()[1]  # bias
    att_weights = np.exp(np.tanh(eij))

    # slice and exclude paddings
    att_weights = att_weights[0][-1 * sent_len:]
    # normalization by z-score
    att_weights = stats.zscore(att_weights)
    att_weights = [abs(att_score) for att_score in att_weights]

    return dict(zip([item for item in single_input[0] if item != 0], att_weights))


def cal_pipeline(input_content, keras_model, input_context=None, input_mode=1, rnn_layer_func_list=None,
                 rnn_layer_func_config=None):
    """To calculate the attention weights

    Args:
        input_content (str): the preprocessed sentence indices
        keras_model ()
        input_context (str): the preprocessed context sentence indices
        input_mode (int): 1 is normal mode, 2 is contextual mode
    """
    if rnn_layer_func_list == None:
        rnn_layer_func_list = get_rnn_func(keras_model, input_mode=input_mode)
    # get attention weights for content
    att_weights = get_att_output(rnn_layer_func_list[0], keras_model, input_content)

    if input_mode == 1:
        # get probabilities of prediction, idx0: 0 (neutral); idx1: 1 (positive); idx2: -1 (negative)
        pred_probs = keras_model.predict_proba(input_content).tolist()[0]
        pred_cls = np.argmax(pred_probs)  # index

        return pred_probs, pred_cls, att_weights, None
    elif input_mode == 2:
        pred_probs = keras_model.predict(
            [input_context, input_content]).tolist()[0]
        sum_probs = sum(pred_probs)
        pred_probs = [itm / sum_probs for itm in pred_probs]

        pred_cls = np.argmax(pred_probs)  # index
        att_weights_context = get_att_output(
            rnn_layer_func_list[1], keras_model,
            input_context, layer_name='context_att'
        )
        return pred_probs, pred_cls, att_weights, att_weights_context
    else:
        return {'error': 'Input Mode should be either content-only(1) or with contxt(2) mode'}
