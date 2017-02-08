import sys
reload(sys)
#sys.setdefaultencoding('utf8')
from sklearn.cross_validation import train_test_split
import multiprocessing
import numpy as np
import nltk
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
import pandas as pd
import sys
# set parameters:
vocab_dim = 100 #the number of vocab_dim must be the same as the number of maxlen 
maxlen = 100
n_iterations = 6  
n_exposures = 10
window_size = 7
cpu_count = multiprocessing.cpu_count()


def loadfile():
    neg=pd.read_excel('neg_train.xls',header=None,index=None)
    pos=pd.read_excel('pos_train.xls',header=None,index=None)
    #print 'pos:', pos[0], neg[0]
    combined=np.concatenate((pos[0], neg[0]))
    #print 'combined:', combined
    #print 'type:', type(combined[0])
    #combined[0] = str(combined[0])
    #print 'type:', type(combined[0])
    #print 'combined:', combined[0]
    y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))
    #print 'y:', y
    return combined,y

def WordTokener(sent):
    sent = nltk.word_tokenize(sent)
    return sent

#create word index, word vector and word indexes in each sentence
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary() #from gensim.corpora.dictionary import Dictionary
        #print 'gensim_dict:', gensim_dict
        gensim_dict.doc2bow(model.vocab.keys(),#the words in model are all more than 10 times
                            allow_update=True) #Convert document (a list of words) into the bag-of-words 
                                               #format = list of (token_id, token_count) 2-tuples
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        #print 'w2indx:', w2indx w2indx: {u'suspenseful': 1, u'four': 2, u'sleep': 3, u'woody': 4, u'hate': 5, 
        #u'poorly': 6, u'relationships': 7, u'whose': 8} u:unique; the number is index
        #print 'w2indx:', w2indx
        w2vec = {word: model[word] for word in w2indx.keys()}#word vector
        #print 'w2vec:', w2vec

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen,padding = "post")
        #Transform a list of sequences (lists of scalars) into a 2D Numpy array of shape
        #maxlen: None or int. Maximum sequence length, longer sequences are truncated 
        #and shorter sequences are padded with zeros at the end.
        return w2indx, w2vec,combined
    else:
        print 'No data provided...'


def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim, #the dimensionality of the feature vectors
                     min_count=n_exposures,#ignore all words with total frequency lower than this.
                     window=window_size, #the maximum distance between the current and predicted word within a sentence
                     workers=cpu_count, #use this many worker threads to train the model (=faster training with multicore machines)
                     iter=n_iterations) #number of iterations (epochs) over the corpus. Default is 5.
    model.build_vocab(combined)
    model.train(combined)
    model.save('Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined

def invert_dict(d):
    return dict([(v, k) for k, v in d.iteritems()])

if __name__ == "__main__":
    combined, y = loadfile()
    for i in range(len(combined)):
        combined[i] = WordTokener(combined[i])
    #qprint 'combined:',combined
    #print 'type of combined:', type(combined[0])
    index_dict, word_vectors, combined = word2vec_train(combined)
    #print 'index_dict:', index_dict
    #print index_dict['four']
    #print 'word_vectors:', word_vectors
    #print 'combined:', combined
    #print word_vectors
    #final_dict = invert_dict(index_dict)
    #for line in combined:
    	#for ele in line:
    		#print final_dict.get(ele),
    #print len(combined)