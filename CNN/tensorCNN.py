"""
tensorCNN.py
description: tensorflow implementation of the two-channel Convolutional Neural Network for
sentiment analysis.
"""

import numpy as np
import tensorflow as tf
import pickle
import os
import time
import datetime
import batchData
from sklearn.cross_validation import KFold

# Set Parameters here ==========================================================
x_dim = 31; # dimension of input vector; padded to 31 words
embedding_size = 41; # dimension of Glove embeddings
embedding_size_wordvec = 100; # Dimension of Word2Vec embeddings
filter_sizes = [3, 4, 5]; # number of words convolutional filter to cover
num_filters = 41; # number of filters per filter size
batch_size = 110; # batch size of data
num_epochs = 11; # number of epochs to use
l2_reg_lambda = 0.5; # l2 regularization coeficient
evaluate_every = 4000; # evaluate model every 1000 step
checkpoint_every = 4000; # save model parameteres every 1000 step
drop_val = 0.5; # dropout factor
clip_norm = 3; # clip weight vector norm
use_crossValidation = False;
continue_train = False;
training_dir = "runs/continuation_of_FullData_training/checkpoints/model-79000" # continue training from specific saved model

"""
Initialize all variables
generate batches of data
start training the defined model
collect training/dev summaries and save checkpoints
"""
def run_CNN():
    if continue_train:
        saver.restore(sess, training_dir)
        print("model restored ...")
    else:
        sess.run(tf.initialize_all_variables());

    # Generate batches
    batches = batchData.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)
    # Training loop. For each batch...
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        # train_step(sess,x, y, keep_prob, x_batch, y_batch, 0.5)
        feed_dict = {
          x: x_batch,
          y: y_batch,
          keep_prob: drop_val
        }
        _, step, summaries, train_loss, train_accuracy = sess.run(
                [train_op, global_step, train_summary_op, loss, accuracy],
                feed_dict)
        time_str = datetime.datetime.now().isoformat()
        # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, train_loss, train_accuracy))
        train_summary_writer.add_summary(summaries, step)

        current_step = tf.train.global_step(sess, global_step)
        if current_step % evaluate_every == 0:
            print("\nEvaluation:")
            # dev_step(sess, x, y, keep_prob, x_dev, y_dev, writer=dev_summary_writer)
            feed_dict = {
              x: x_dev,
              y: y_dev,
              keep_prob: 1.0
            }
            step, summaries, train_loss, train_accuracy = sess.run([global_step, dev_summary_op, loss, accuracy],feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, train_loss, train_accuracy))
            dev_summary_writer.add_summary(summaries, step)
            # train_accuracy = accuracy.eval(feed_dict, session=sess);
            # print("accuracy: ", train_accuracy)
            print("")
        if current_step % checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))


# load embeddings ==========================================================

# Glove embeddings
embeddings = np.load("../modelData/embeddings_nostop_40_sum_pad.npy");
embeddings = np.array(embeddings, np.float32);

# Word2Vec Embeddings
embeddings_word2vec = np.load("../word2vec_data/emb_100.npy");
embeddings_word2vec = np.array(embeddings_word2vec, np.float32);

# Placeholders =============================================================
x = tf.placeholder(tf.int32, [None, x_dim], name="input_x_1");
y = tf.placeholder(tf.float32, [None, 2], name="input_y");

# variables
l2_loss = tf.constant(0.0)

# Embedding layer ==========================================================
with tf.name_scope("embedding_glove"):
    W_embedding = tf.Variable(embeddings, name="W_embedding_glove")

    embedded_chars = tf.nn.embedding_lookup(W_embedding, x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    embedded_chars_expanded = tf.cast(embedded_chars_expanded, tf.float32);


with tf.name_scope("embedding_word2vec"):
    W_embedding_word2vec = tf.Variable(embeddings_word2vec, name="W_embedding_word2vec")

    embedded_chars_word2vec = tf.nn.embedding_lookup(W_embedding_word2vec, x)
    embedded_chars_expanded_word2vec = tf.expand_dims(embedded_chars_word2vec, -1)
    embedded_chars_expanded_word2vec = tf.cast(embedded_chars_expanded_word2vec, tf.float32);

# Convolutional layer for Glove Embeddings Channel =============================
# concatinated = []

pooled_outputs = [];
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, dtype=tf.float32), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters], dtype=tf.float32), name="b")

        conv = tf.nn.conv2d(
            embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(conv + b, name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, x_dim - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooled_outputs.append(pooled)
        # concatinated.append(pooled)

# Convolutional layer for Word2Vec Embeddings Channel ==========================
pooled_outputs_word2vec = [];
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        filter_shape = [filter_size, embedding_size_wordvec, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, dtype=tf.float32), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters], dtype=tf.float32), name="b")

        conv = tf.nn.conv2d(
            embedded_chars_expanded_word2vec,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(conv + b, name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, x_dim - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooled_outputs_word2vec.append(pooled)
        # concatinated.append(pooled)

# Combine all the pooled features ==============================================
num_filters_total = num_filters * len(filter_sizes)

h_pool = tf.concat(3, pooled_outputs); # 3 for one channel implementation
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

h_pool_word2vec = tf.concat(3, pooled_outputs_word2vec); # 3 for one channel implementation
h_pool_flat_h_pool_word2vec = tf.reshape(h_pool_word2vec, [-1, num_filters_total])

# experimental
# h_conc_pool = tf.concat(3, concatinated)
# h_conc_pool_flat = tf.reshape(h_conc_pool, [-1, num_filters_total]);

# Combine Glove and Word2Vec channels
h_pool_additive = h_pool_flat_h_pool_word2vec + h_pool_flat;

# Add dropout ==================================================================
keep_prob = tf.placeholder(tf.float32, name="kee_prob")
with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_additive, keep_prob)

# Final scores and predictions =================================================
with tf.name_scope("output"):
    W = tf.Variable(tf.truncated_normal([num_filters_total, 2], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")

# CalculateMean cross-entropy loss ============================================
with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(scores, y)
    loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss;

# Accuracy =====================================================================
with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

# Data Preparatopn
# ==============================================================================

# Load data
print("Loading data...")
x_file = "../twitter-datasets/trainX_combined_partial20.npy"
y_file = "../twitter-datasets/labels_partial20.npy"
if use_crossValidation:
    X_data, Y_data = batchData.load_data(x_file,y_file,useEvaluationSet=False);
else:
    x_train, y_train, x_dev, y_dev = batchData.load_data(x_file, y_file,useEvaluationSet=True);
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training =====================================================================
sess = tf.Session();

# Define Training procedure
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-3) # 1e-3

grads_and_vars = optimizer.compute_gradients(loss)
# enforcing l2 constraing (for overfitting)
for i, (grad, var) in enumerate(grads_and_vars):
    if grad is not None:
        grads_and_vars[i] = (tf.clip_by_norm(grad, clip_norm), var);

train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# Summary ======================================================================
# Output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))

# Summaries for loss and accuracy
loss_summary = tf.scalar_summary("loss", loss)
acc_summary = tf.scalar_summary("accuracy", accuracy)

# Train Summaries
train_summary_op = tf.merge_summary([loss_summary, acc_summary])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

# Dev summaries
dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.all_variables())


# Cross validation =============================================================
if use_crossValidation:
    kf = KFold(X_data.shape[0], n_folds=5, shuffle=True)
    for k,(train_ind, test_ind) in enumerate(kf):
        print(k, "th iteration (cross validation)!")
        x_train, x_dev = X_data[train_ind], X_data[test_ind]
        y_train, y_dev = Y_data[train_ind], Y_data[test_ind]
        run_CNN();
else:
    run_CNN();
