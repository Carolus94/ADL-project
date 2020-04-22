# Tosin Adewumi

"""
First run:
python -m gensim.scripts.make_wiki

"""

import re
import time
from nltk.corpus import stopwords
from nltk import word_tokenize
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.corpora import WikiCorpus, MmCorpus
import logging
import os
from multiprocessing import cpu_count


LANGUAGE = "english"
DIRECTORY = "data" # /home/shared_data or data BY CREATING A DATA FOLDER & PUT ONE DATASET AT A TIME, i.e. WHEN FINISHED WITH 3.9 FOR ALL DIMENSIONS THEN DELETE & PUT 4.9



class ReadLinebyLine():
    def __init__(self, corpus_location, language):
        """

        :param corpus_location:
        """
        self.directory = corpus_location
        assert isinstance(self.directory, str), "method requires string corpus location"
        self.stop_words = set(stopwords.words(language))    # no support for yoruba
        self.stop_words.update(['.',',',':',';','(',')','#','--','...','"','_','|','Â»','%','[',']','{','}'])


    def __iter__(self):
        """

        :return:
        """
        for fname in os.listdir(self.directory):                                        # go over all files in directory
            for line in open(os.path.join(self.directory, fname), encoding="utf8"):     # read file line by line in utf8
                line = line.lower()
                line = re.sub("<.*?>", "", line)            # removes html tags
                tokenized_text = word_tokenize(line)
                if len(tokenized_text) == 0:                # replace empty lists with BLANKTOKEN
                    tokenized_text = ["BLANKTOKEN"]
                tokenized_text = [nonum for nonum in tokenized_text if not nonum.isnumeric()]   # remove numbers
                yield [w for w in tokenized_text if w not in self.stop_words]   # returns memory-efficient generator


if __name__ == "__main__":
    """
    Dim sizes for experiment:
    # size: 200, 300, 400, 500, 600, 700, 800
    """
    sizes = [200, 300, 400, 500, 600, 700, 800]
    for x in range(len(sizes)):
        size = sizes[x]
        # logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=1)
        starttime = time.time()
        processed_corpus = ReadLinebyLine(DIRECTORY, LANGUAGE)     # memory-efficient iterator for regular files
        model = Word2Vec(processed_corpus, min_count=5, size=size, workers=cpu_count(), window=8, sg=0, hs=0, negative=5, iter=10, compute_loss=True) # CHANGE SIZE
        time_elapsed = time.time() - starttime
        print("Vocab", len(model.wv.vocab))
        print("Time elapsed", time_elapsed)
        savename = "w2v_m5_s"+str(sizes[x])+"_w8_s0_h0_n5_i10"
        model.save(savename) # CHANGE MODEL NAME FOR EVERY OUTPUT
        OUTPUT_FILE1 = "out_" + savename+".txt" # CHANGE FILENAME FOR EVERY OUTPUT
        # evaluate on word analogies - modern version of accuracy
        human_word_sim = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
        analogy_scores = model.wv.evaluate_word_analogies(datapath("questions-words.txt"))
        with open(OUTPUT_FILE1, "w+") as f:
            s = f.write("Vocab: " + str(len(model.wv.vocab)) + "\n" "Training loss: " +
                        str(model.get_latest_training_loss()) + "\n" "Time elapsed: " + str(time_elapsed) +
                        "\n" "Analogy Scores: " + str(analogy_scores) + "\n" "Human Word Similarity: " + str(human_word_sim))
