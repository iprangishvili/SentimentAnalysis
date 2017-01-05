import gensim, logging
import numpy as np
import pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def main():
    # Load vocabulary
    with open('modelData/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    # Load data
    x_train = [];
    for fn in ['twitter-datasets/train_pos_full.txt', 'twitter-datasets/train_neg_full.txt']:
        with open(fn) as f:
            for line in f:
                lineData = [];
                sline = line.strip().split();
                for word in sline:
                    word_id = vocab.get(word, -1);
                    if word_id != -1:
                        lineData.append(str(word_id)); # change here
                x_train.append(lineData);

    # Train word2vec model
    model = gensim.models.Word2Vec(size=100, min_count=1)
    model.build_vocab(x_train,keep_raw_vocab=True)
    model.train(x_train)

    # Convert word2vec embedding to numpy array representation
    vocab_size = 21161;
    word2vec_emb = [];
    for i in range(vocab_size):
        word2vec_emb.append(model[str(i)])
    zero_pad = [0 for i in range(100)]
    word2vec_emb.append(zero_pad)
    word2vec_emb = np.array(word2vec_emb);

    print(word2vec_emb.shape)
    # save embedding as numpy array
    np.save("word2vec_data/emb_100", word2vec_emb)

if __name__ == "__main__":
    main()
