import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'


import keras
from keras.models import Sequential
import tensorflow as tf


from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout
from keras.layers import Reshape
from keras.models import load_model

#from keras.utils.training_utils import multi_gpu_model
#from hyperparameterConfig import Hyperparameters
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import Input, dot
from keras.utils import np_utils
from keras import regularizers

import time
import numpy as np
import pandas as pd
import glob
import pdb
EMBEDDING_DROPOUT =  .2
#EMBED_SIZE = 50
#NEW_DRUGS = 1270 #100

import sys
import pickle
import time

#embed_indications.run("skips-60-480-0.6-tf/","vocab_drugs.pkl", "skips-60-480-0.6/skipdat.*",settings=settings)
def run(fnamein, vocab, skips, onlyDx=False,do_early_stopping=True,
        settings={}):

    (fname, settingsList) = model_util.make_fname(fnamein, settings)

    if not 'dropout' in settings:
        settings['dropout'] = EMBEDDING_DROPOUT
    if not 'momentum' in settings:
        settings['momentum'] = .9
    if not 'cuda' in settings:
        settings['cuda'] = "0"
    if not 'learning_rate' in settings:
        settings['learning_rate'] = .0001
    if not 'decay' in settings:
        settings['decay'] = 0
    if not 'batchsize' in settings:
        settings['batchsize'] = 500
    if not 'opt' in settings:
        settings['opt'] = 'sgd'
    if not 'totdat' in settings:
        settings['totdat'] = 100000000
    if not 'numep' in settings:
        settings['numep'] = 20
    if not 'nesterov' in settings:
        settings['nesterov'] = False

    eltfreq, newdrugs = pickle.load(open(vocab,'rb'))
    eltfreq = eltfreq.rename({'vid':'VOCAB'},axis=1)
    VOCAB_SIZE = int(eltfreq['VOCAB'].max()) + 1 ## for zero
    eltfreq.to_csv("forcallback.txt",sep="\t")
    NEW_DRUGS = newdrugs.shape[0]
    print("vocab_size = ", VOCAB_SIZE, " new drugs = ",NEW_DRUGS)
    os.environ["CUDA_VISIBLE_DEVICES"]=settings['cuda']

    step_per_ep = int(settings['totdat']/settings['batchsize']) 
    val_step =  int(settings['valdat']/settings['batchsize']) 
    print(fname)
    #pdb.set_trace()
    print( "\n".join(settingsList))
    model_util.tf_sess()            
    model = Sequential()
    if 'resumefile' in settings:
        model = load_model(settings['resumefile']) 
    else:
        with tf.device('/gpu:0'):
            embedding = Embedding(VOCAB_SIZE, settings['embed_size'], name='embed1', input_length=1)
            model.add(embedding)
            model.add(Reshape((settings['embed_size'],)))
            dropoutLayer = Dropout(settings['dropout'])
            model.add(dropoutLayer)

            model.add(Dense(NEW_DRUGS, activation='softmax',
                            kernel_regularizer=regularizers.l2(settings['L2']))) #"softmax"))

            opt = get_opt(settings)
            model.compile(loss='categorical_crossentropy', optimizer =opt,metrics=['accuracy']) #'adam'
            ###########
            ###########


    files = glob.glob(skips)
    if 'max_files' in settings:
        files = files[:settings['max_files']]
    test_files = np.random.choice(files, int(.1*len(files)))
    files = list(set(files) - set(test_files))
    gTrain = generate_generic(files,settings['batchsize'], NEW_DRUGS)
    gValid = generate_generic(test_files,settings['batchsize'], NEW_DRUGS)
    checkpoint = ModelCheckpoint(filepath=fname + "-{epoch:03d}.h5")

    early = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=1, mode='auto')    
    time_callback = TimeHistory()
    val_callback = ValGo()
    callbacklist = [checkpoint, time_callback, val_callback]
    if do_early_stopping:
        callbacklist.append(early)
    history = model.fit_generator(generator=gTrain, verbose=1,
            validation_data = gValid, validation_steps=val_step,
                                  steps_per_epoch=step_per_ep, epochs=settings['numep'],
                                  callbacks=callbacklist) #, early, class_weight=, callbacks=[mycallback,early_stopping])
    end_save(model, history, fname, settingsList, time_callback)
    
    f = open(fname + ".val",'wb')
    pickle.dump((val_callback.valid_examples, val_callback.results),f)
    f.close()

    

def make_fname(fnamein, settings):
    return (fnamein + ".".join( ['{:1.2e}'.format(v) if type(v)==np.float64 else str(v) for k, v in settings.items() if not k in ['cuda','resumefile']]),
           [k + "=" + str(v) #('{:1.2e}'.format(v) if type(v)==np.float64 else str(v))
                    for k,v in settings.items()
                    if not k=="cuda" ])

def get_opt(settings):
    opt = settings['opt']
    if settings['opt']=='sgd':
        opt = optimizers.SGD(lr=settings['learning_rate'], decay=settings['decay'], momentum=settings['momentum'], nesterov=True)
    elif 'learning_rate' in settings:
        if settings['opt']=='adam':
            opt = optimizers.Adam(lr=settings['learning_rate'])
    return opt

def end_save(model, history, fname, settingsList, time_callback):
    model.save(fname + '.model.h5')
    his = open(fname + ".history.csv",'w')
    his.write('\t'.join(settingsList))
    df = pd.DataFrame(history.history)
    times = time_callback.times
    df['times'] = times
    best = list(df.loc[df['val_loss']==df['val_loss'].min(),:].index)[0]
    os.rename("{:s}-{:03d}.h5".format(fname, best), "{:s}-best.h5".format(fname))
    df.to_csv(his)
    his.close()
            

def tf_sess():

    config = tf.ConfigProto()

    config.allow_soft_placement=True
    config.gpu_options.allow_growth=True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    K.set_session(sess)

def generate_generic(files, batch_size, num_classes,  onlyDx=False):
    #fdo = files[22]
    while True:
        #print("RESTART GEN!")
        np.random.shuffle(files)
        for fi in files:
            #print(fi)
            #fi = 'sample3/skipdat.8.32' ## OVERFIT #fdo #
            dat = pd.read_csv(fi,sep="\t",header=None).values
            i0 = 0
            #i1 = batch_size
            nrow = dat.shape[0]
            ix = np.arange(nrow)
            np.random.shuffle(ix)
            dat = dat[ix,:]
            #print(dat[:5,:])
            #print(max(dat[:,1]))
            labs = np_utils.to_categorical(dat[:,1], num_classes=num_classes)
            while i0 < nrow:
                rowend = min(i0 + batch_size,nrow)
                yield (dat[i0:rowend,0], labs[i0:rowend,:])
                # batch[:,1])
                i0 += batch_size

class ValGo(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.eltfreq = pd.read_csv("forcallback.txt",sep="\t")
        self.valid_examples = []
        for typ in [i for i in set(self.eltfreq['type']) if not pd.isnull(i)]:
            options = list(self.eltfreq.loc[self.eltfreq['type']==typ,:].sort_values('ct',ascending=False)[:100]['VOCAB'])
            #print(typ, options)
            self.valid_examples.extend(list(np.random.choice(options,4)))
        self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
        self.results = []
        
    def on_epoch_end(self, batch, logs={}):
        emb = tf.Variable(self.model.get_layer("embed1").get_weights()[0])
        norm = tf.sqrt(tf.reduce_sum(tf.square(emb), 1, keep_dims=True))
        normalized_embeddings = emb / norm
        valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, self.valid_dataset)
        self.similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)
        
        sim =  K.eval(self.similarity)
        self.results.append(sim)
        print("\n________")
        for i, v in enumerate(self.valid_examples):
            vname = self.eltfreq.loc[self.eltfreq['VOCAB']==v,'code'].item()
            top_k = 6
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            try:
                nnames = [self.eltfreq.loc[self.eltfreq['VOCAB']==k,'code'].item()
                      if (self.eltfreq['VOCAB']==k).sum()==1 else "EMPTY"
                      for k in nearest]
                #print("; ".join(nnames))
                print("Nearest to " + vname + ": " + "; ".join(nnames))
            except TypeError:
                print("Typerror!", vname)
                print(nearest)

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)





            
import argparse
#bash run_emb.sh -l2 .001 -lr .0001 -es 50 -ep 198 -st 0 -skips skips-60-480-0.6  -max_files 2400 -voc vocab_drugs.pkl
if __name__ == "__main__":
    #huh()

    parser = argparse.ArgumentParser(description='Setup.')
    parser.add_argument('-l2', dest='l2', 
                     default=10**(-1*(np.random.rand(1)*3 + 1)[0]),
                        help='sum the integers (default: find the max)',type=float)
    parser.add_argument('-cuda', dest='cuda', 
                        default='0')
    parser.add_argument('-es', dest='embed_size', default=10,type=int)
    parser.add_argument('-lr', dest='learning_rate',
                        default=10**(-1*(np.random.rand(1)*4 + 1.5)[0]),type=float)
    parser.add_argument('-st', dest='earlystopping',
                        default=0,type=int)
    parser.add_argument('-ep', dest='numep',
                        default=20,type=int)
    parser.add_argument('-bs', dest='batchsize',
                        default=200,type=int)
    
    parser.add_argument('-resume', dest='resumefile',
                        default="NO",type=str)
    parser.add_argument('-skips', dest='skippath',
                        default='groupvoc/skipdat.*',type=str)
    parser.add_argument('-save', dest='savepath',
                        default='',type=str)
    parser.add_argument('-max_files', dest='max_files',type=int)
    
    voc = "../../data/clid.vi.allvocab.pkl"     # voc = "../../data/clid.uniqvoc.pkl"
    parser.add_argument('-voc', dest='vocab',type=str,default=voc)
    args = parser.parse_args()
    #learning_rate = sys.argv[3]
    prefix = "embgr/"


    settings = {'opt':'adam', 'L2':args.l2, 'cuda':'0','batchsize':args.batchsize,
                'numep':args.numep,'totdat':5000000,'valdat':3000000,
                'embed_size':args.embed_size}
    if 'max_files' in args:
        print("max_files = ", args.max_files)
        settings['max_files'] = args.max_files
    if args.learning_rate > 0:
        settings['learning_rate']  = args.learning_rate
    if not args.resumefile == "NO":
        settings['resumefile']  = args.resumefile
    saveto = args.savepath
    if not saveto:
        saveto = args.skippath.strip("/") + "-tf/"
    if not saveto.endswith("/"):
        saveto += "/"
    if not os.path.exists(saveto):
        os.mkdir(saveto)
    print(args)

    run(saveto,args.vocab, args.skippath + "/skipdat.*",
        do_early_stopping = args.earlystopping==True,
                settings=settings)




def get_dfs(fname, drugdf,embonly=True, epoch=-1):
    toopen = '{:s}-{:03d}.h5'.format(fname,epoch ) if epoch > 0 else '{:s}-{:s}.h5'.format(fname, 'best')
    if not os.path.exists(toopen):
        return None
    e0 = load_model(toopen) 

    emb = pd.DataFrame(e0.get_layer('embed1').get_weights()[0])
    emb = (emb.transpose()/ ((emb**2).sum(axis=1)**.5 )).transpose() #.var(axis=1)
    if embonly:
        return emb
    namp = e0.predict(np.arange(emb.shape[0]))
    if namp.shape[1] > drugdf.shape[0]:
        namp = namp[:,:drugdf.shape[0]]
    namp = pd.DataFrame(namp, columns = drugdf['name']) # #ef['eltid']))
    wt = e0.layers[3].get_weights()[0]
    wt = wt/((wt**2).sum(axis=0)**.5 )
    if wt.shape[1] > drugdf.shape[0]:
        print("trim WT",wt.shape[1],drugdf.shape[0])
        wt = wt[:,:drugdf.shape[0]]
    d0 = pd.DataFrame(wt,columns=drugdf['name'])
    return emb, d0, namp
