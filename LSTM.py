import numpy as np 
import theano as theano
import theano.tensor as T 

class LSTM:

    def __init__(self, word_dim, hidden_dim):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim 
        params = {}
        params[
        params["W_f"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        params["W_i"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        params["W_c"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        params["W_o"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        params["B_f"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim))
        params["B_i"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim))
        params["B_c"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim))
        params["B_o"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim))
        params["lr"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (2, hidden_dim)) 

