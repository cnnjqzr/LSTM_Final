from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne
import nltk


N_hidden = 10

Learning_rate = 0.005

Grad_clip = 100

Epoch = 1000

N_epoch = 50

def Preprocessing(File_name):
    File = open(File_name, "r+")
    Preprocessed = []
    for line in File:
        Preprocessed.append(Word_tokenize(line))
    return Preprocessed

def Word_tokenize(Line):
    Line = nltk.word_tokenize(Line)
    return Line

def Create_dict(List):
    Dict = {}
    Pointer = 1
    for line in List:
        for item in line:
            if item not in Dict.keys():
                Dict[item] = Pointer
                Pointer += 1
    return Dict
def Word2Vec(List,dict):
    New_List = []
    Max_len = 0
    N_batch = len(List)
    for line in List:
        New_line = []
        if len(line) > Max_len:
            Max_len = len(line)
        for item in line:
            New_item = dict[item]
            New_line.append(New_item)
        New_List.append(New_line)
    return New_List,Max_len,N_batch
def Add_zero(List, Max_len):
    for line in List:
        if len(line) < Max_len:
            for i in range(Max_len-len(line)):
                line.append(0)
    return List
def Get_Label(Source_list):
    Label = []
    List_without_label = []
    for line in Source_list:
        Label_list = [line[0],1-int(line[0])]
        Label.append(Label_list)
        List_without_label.append(line[1:])
    return Label,List_without_label
def Set_data(Input,Label,N_batch,Seq_length):
    X_input = np.asarray(Input)
    X = np.expand_dims(Input, axis=-1)    
    X = np.concatenate([X,
                        np.zeros((N_batch, Seq_length, 1))],axis=-1)
    Y = np.asarray(Label)
    mask = np.zeros((N_batch, Seq_length))
    for n in range(N_batch):
        Line = X_input[n]
        for j in range(Seq_length):
            if Line[j]!= 0:
                mask[n][j] = 1
            else:

                mask[n][j] = 0
    return X.astype(theano.config.floatX) , Y.astype(theano.config.floatX) , mask.astype(theano.config.floatX)

def main():
    PPP = Preprocessing("train.txt")
    Label , final_file = Get_Label(PPP)
    Dict = Create_dict(final_file)
    Vec_list , Seq_length , N_batch= Word2Vec(final_file,Dict)
    Final_list = Add_zero(Vec_list,Seq_length)
    X,Y,Mask = Set_data(Final_list,Label,N_batch,Seq_length)
    l_in = lasagne.layers.InputLayer(shape=(N_batch, Seq_length,2))
    l_mask = lasagne.layers.InputLayer(shape=(N_batch,Seq_length))
    gate_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),b=lasagne.init.Constant(0.))
    cell_parameters = lasagne.layers.recurrent.Gate( W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),W_cell=None, b=lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.tanh)
    l_lstm = lasagne.layers.LSTMLayer(l_in, N_hidden , grad_clipping = Grad_clip , mask_input = l_mask , 
        ingate=gate_parameters, forgetgate=gate_parameters, cell=cell_parameters, outgate=gate_parameters,
        only_return_final = True , learn_init = True )
    l_rnn = lasagne.layers.RecurrentLayer(
        l_in, N_hidden, mask_input=l_mask, grad_clipping=Grad_clip,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)
    l_out = lasagne.layers.DenseLayer(l_lstm, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    network_output = lasagne.layers.get_output(l_out)
    target_values = T.matrix('target_output')
    cost = lasagne.objectives.categorical_crossentropy(network_output, target_valuesc).mean()
    #cost = T.nnet.categorical_crossentropy(predicted_values,target_values).mean()
    all_params = lasagne.layers.get_all_params(l_out)
    updates = lasagne.updates.adagrad(cost, all_params, Learning_rate)
    train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                            cost, updates=updates)
    compute_cost = theano.function(
        [l_in.input_var, target_values, l_mask.input_var], cost)
    X_val, y_val, mask_val = X,Y,Mask
    print("Training ...")
    try:
        for epoch in range(100000000):
            for _ in range(1):
                va = train(X, Y, Mask)
            cost_val = compute_cost(X_val, y_val, mask_val)
            print("Epoch {} validation cost = {}".format(epoch, cost_val))
    except KeyboardInterrupt:
        pass
main()
    