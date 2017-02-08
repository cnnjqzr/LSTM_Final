import numpy as np
import theano as theano
import theano.tensor as T
import operator
import time
import sys
import os
from datetime import datetime
import preprocessing as pp 
class RNNTheano:
    
    def __init__(self, word_dim, hidden_dim=10, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        LR = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),(2, word_dim))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))  
        # LR is the matrix which control the final output of segmental
        self.LR = theano.shared(name = 'LR', value=LR.astype(theano.config.floatX))    
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def print_LR (self):
    	print self.LR.get_value()

    def __theano_build__(self):
        U, V, W, LR= self.U, self.V, self.W, self.LR
        x = T.dmatrix('x')
        y = T.ivector('y')
        def forward_prop_step(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U.dot(x_t) + W.dot(s_t_prev))
            o_t = T.nnet.softmax(V.dot(s_t))
            return [o_t[0], s_t]
        [o,s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None,dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[U, V, W],
            truncate_gradient=self.bptt_truncate,
            strict=True)
        prediction = T.mean(o)#mean of all output 
        #prediction = T.argmax(o, axis=1)
        out_put = T.nnet.softmax(self.LR.dot(prediction))# output = Softmax(LR*prediction) [p(pos),p(neg)]
        o_error = T.sum(T.nnet.categorical_crossentropy(out_put, y))
        #prediction = T.argmax(o, axis=1)
        #o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        # Gradients
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)
        dLR = T.grad(o_error,LR)
          
        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.ce_error = theano.function([x, y], o_error)
        
        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [], 
                      updates=[(self.U, self.U - learning_rate * dU),
                              (self.V, self.V - learning_rate * dV),
                              (self.W, self.W - learning_rate * dW),
                              (self.LR, self.LR - learning_rate * dLR)                  
                              ])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(X,y) for y in Y])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)   

def index_to_matrix (index,dict,x_line,t):
    count = 0 
    for i in x_line:
        word = dict.get(i)
        x_line = index.get(word)
        #if x_line is not None:
            #x_matrix.append(x_line.tolist())
            #x_list = x_line.tolist()
            #x_matrix.append(x_list)
        #print x_line
    r = np.random
    r.seed(t)
    D = r.randn(30,100).tolist()
    return D   
def invert_dict(d):
    return dict([(v, k) for k, v in d.iteritems()]) 
def train_with_sgd(model, X_train, y_train,index,dic, learning_rate=0.05, nepoch=1, evaluate_loss_after=5 ):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = 0
            t = 0
            #model.print_LR()
            for x_line in X_train:
                x_matrix = index_to_matrix(index,dic,x_line,t) 
                loss += model.calculate_loss(x_matrix, y_train)
            losses.append((num_examples_seen, loss/len(y_train)))
           # model.print_U()
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss/len(y_train))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            #save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
           #  model.print_LR()
        # For each training example...
        for i in range(len(y_train)): # how many vectors should we process
            # One SGD step
            X_matrix = index_to_matrix(index,dic,X_train[i],i)
            model.sgd_step(X_matrix, y_train[i],learning_rate)
            num_examples_seen += 1
combined, y = pp.loadfile()
for i in range(len(combined)):
    combined[i] = pp.WordTokener(combined[i])
index_dict, word_vectors, combined = pp.word2vec_train(combined)
model = RNNTheano(100)
#N = len(combined)
N = 100
feats = 100
rng = np.random
rng.seed(123)
index_dict = invert_dict(index_dict)
D = rng.randint(low = 0 , high = 2 , size = (N , 2)).tolist()
train_with_sgd(model, combined[:100], D, word_vectors,index_dict,nepoch=100, learning_rate=0.005)