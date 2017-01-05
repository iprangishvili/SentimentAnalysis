"""
run.py
description: generates predictions of test data and creates a submission file
based on the specified CNN model checkpoint.
"""
import tensorflow as tf
import numpy as np

# Parameters
# ==================================================

print("Loading data...")
test_X = np.load("../twitter-datasets/testX.npy");

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = "1466591113/checkpoints/model-136000"; # model to use for prediction
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x_1").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("kee_prob").outputs[0]
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        test_predictions = sess.run(predictions, {input_x: test_X, dropout_keep_prob: 1.0})

# Print accuracy
print(test_predictions);
res_test = np.array(test_predictions, dtype=np.int32);
f_out = open("../submission_CNN.csv", "w");
f_out.write("Id,Prediction");
for i, pred_label in enumerate(res_test):
    f_out.write("\n");
    if pred_label == 0:
        pred_str = "-1";
    else:
        pred_str="1";
    f_out.write(str(i+1) + "," + pred_str);
f_out.close();
