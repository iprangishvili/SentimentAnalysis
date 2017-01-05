"""
kerasLSTMGRU.py

description:
This program implements a sentiment analysis model. It builds a
Neural Network model with: input layer, followed by embedding layer initialized
with pre-trained Word2Vec embeddings, LSTM layer for sentence composition,
Forward Gated and backward Gated NN, and a Softmax layer.
Model takes as an input pre-processed data. The input sentances are already
tokenized, and converted to vocabulary representation to be able to feed into
embedding layer.

implemented using keras. (backend tensorflow)
"""
import numpy as np
from keras.models import Sequential, Graph
from keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Activation, Merge
from keras.layers import Embedding
from keras.layers import LSTM, GRU
from sklearn.cross_validation import train_test_split
from keras.regularizers import l2, activity_l2

np.random.seed(10)  # for reproducibility

# Set model Parameters here ====================================================

maxlen = 31 # padded length of each input sentence
embedding_size = 100 # dimension of Word2Vec embeddings
vocab = 21162 # size of the vocabulary
lstm_output_size = 100 # output dimension of LSTM cell
GRU_output_size = 40 # output dimension of GRU cell
batch_size_tr = 110 # batch size for training
nb_epoch = 2 # number of epochs
load_model = False; # Load waights of saved model
current_model_name = 'LSTM_GRU_weights_ep2_100.h5' # name of the current model to be saved as.
evaluate_test = False; # Load already saved model and make prediction. (without training)


# load Word embedding ==========================================================
embeddings_word2vec = np.load("../word2vec_data/emb_100.npy");
embeddings_word2vec = np.array(embeddings_word2vec, np.float32);

# LSTM-GRU model definition ====================================================
print('Build model...')

# initiate a model
model = Graph()
# add an input
model.add_input(name='input', input_shape=(maxlen,), dtype='int')
# add an Embedding Layer initialized with pre-trained Word2Vec embeddings
model.add_node(Embedding(vocab, embedding_size, weights=[embeddings_word2vec], input_length=maxlen),
                    name="embedding", input="input")
# add a LSTM layer
model.add_node(LSTM(lstm_output_size, return_sequences=True), name="LSTM_cell", input="embedding")
# add forward GRU
model.add_node(GRU(GRU_output_size), name='gru_forward', input="LSTM_cell")
# add backwards GRU
model.add_node(GRU(GRU_output_size, go_backwards=True), name='gru_backward', input="LSTM_cell")
# concatinate two GRU layers and apply a dropout layer with probability = 0.5 (for overfitting)
model.add_node(Dropout(0.5), name="GRU_outputs", inputs=["gru_forward", "gru_backward"])
# add fully connected layer with l2 rehularixations applied
model.add_node(Dense(2, W_regularizer=l2(0.01), b_regularizer=l2(0.01)),
                    name="full_con", input="GRU_outputs")
# add an activation function (softmax)
model.add_node(Activation('softmax'), name="prob", input="full_con")
# add output layer
model.add_output(name='pred', input='prob')
# compile model with binary_crossentropy as a loss functions
# and adam optimizer
model.compile(loss={'pred':'binary_crossentropy'},
              optimizer='adam',
              metrics=['accuracy'])

# save the model architecture
json_string = model.to_json()
open('LSTM_GRU_architecture.json', 'w').write(json_string)

# model trainin/evaluation =====================================================
if evaluate_test==False:

    print("Loading training data...")
    x_file = "../twitter-datasets/trainX_combined.npy"
    y_file = "../twitter-datasets/labels.npy"

    # combined (positive, negative) training data, Full set
    # words are represented by vocabulary ids for word embedding lookup
    x_d = np.load(x_file)
    # combined (positive, negative) training data, Full set. Labels
    y_d = np.load(y_file)

    print("Train/Test split")
    X_train, X_test, y_train, y_test = train_test_split(x_d, y_d,test_size=0.05, random_state=10)

    # Continue training of the model from specified model checkpoint
    if load_model == True:
        model_checkpoint_name = 'LSTM_GRU_weights_ep2_100.h5' # previously saved model to continue traing from.
        print("Loading model: ", model_checkpoint_name)
        model.load_weights(model_checkpoint_name); # load model weights for continuing training

    print('Train...')
    model.fit({'input': X_train, 'pred': y_train},
                batch_size=batch_size_tr, nb_epoch=nb_epoch,
                validation_data=({'input': X_test, 'pred': y_test}))

    model.save_weights(current_model_name) # save the model weights after training
else:
    print("Loading evaluation data...")
    test_X = np.load("../twitter-datasets/testX.npy")
    current_model = 'LSTM_GRU_weights_ep2_100.h5' # model name to be used for evaluation
    model.load_weights(current_model) # load model weights
     # make prediction
    X_pred = np.array(model.predict({'input': test_X})['pred'])
    pred = np.argmax(X_pred, axis=1)
    res_test = np.array(pred, dtype=np.int32);
    # create a submission file
    f_out = open("../submission_LSTM_GRU.csv", "w");
    f_out.write("Id,Prediction"); # add a header
    for i, pred_label in enumerate(res_test):
        f_out.write("\n");
        if pred_label == 0:
            pred_str = "-1";
        else:
            pred_str="1";
        f_out.write(str(i+1) + "," + pred_str);
    f_out.close();
