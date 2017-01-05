"""
pickle_vocab.py
description: read vocab_cut.txt file generate from shell scripts provided
and save it as a pickle file. Add vocabulary ID for <PAD/> for later use.
(For Neural Network models where sentences are padded to fixed length with <PAD/>)
"""
import pickle

def main():
    vocab = dict()
    with open('../shellScripts/vocab_cut.txt') as f:
        line_counter = 0;
        for line in f:
            vocab[line.strip()] = line_counter
            line_counter += 1;
        vocab["<PAD/>"] = line_counter;

    with open('../modelData/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
