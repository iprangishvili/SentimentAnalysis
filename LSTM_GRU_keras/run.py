from keras.models import model_from_json
import numpy as np

def predict():
    # load model architecture
    model = model_from_json(open('LSTM_GRU_architecture.json').read())
    # load model weights
    model.load_weights('LSTM_GRU_weights_ep2_100.h5')
    # load test data
    test_X = np.load("../twitter-datasets/testX.npy")

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

if __name__ == "__main__":
    predict()
