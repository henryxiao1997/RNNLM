# -- coding:utf-8

from torch.utils.data import Dataset
from torch.autograd import Variable
import tensorflow as tf
import torch
import numpy as np

class LMDataset(Dataset):
    def __init__(self, vocabPath, dataPath, seqLength):
        super().__init__()
        self.wordlist, self.word2id, self.id2word = self.LoadVocab(vocabPath)
        self.seqLength = seqLength
        self.data = self.LoadData(dataPath)
        pass

    def __getitem__(self, index):
        x = self.data[0][index]
        xt = Variable(torch.from_numpy(np.array(x, dtype=np.int64)))
        y = self.data[1][index]
        yt = Variable(torch.from_numpy(np.array(y, dtype=np.int64)))
        return (xt, yt)

    def __len__(self):
        length = len(self.data[0])
        return length
    
    def LoadVocab(self, vocabPath):
        wordlist = list()
        word2id = dict()
        id2word = dict()
        with open(vocabPath, 'r') as fIn:
            for item in fIn.readlines():
                wordlist.append(item.strip())
        wordlist.append("<eos>")
        wordlist.sort()
        for id, word in enumerate(wordlist):
            word2id[word] = id
            id2word[id] = word
        return wordlist, word2id, id2word

    def SplitListBySeqLength(self, dataListIn):
        seqLists = list()
        seqNum = len(dataListIn) // self.seqLength
        for i in range(seqNum):
            seqList = dataListIn[i*self.seqLength:(i+1)*self.seqLength]
            seqLists.append(seqList)
        return seqLists

    def LoadData(self, dataPath):
        with tf.io.gfile.GFile(dataPath, "r") as fIn:
            data = fIn.read().replace("\n", " <eos> ").split()
            data_ids = \
                [self.word2id[word] for word in data if word in self.word2id.keys()]
            dataX = data_ids
            dataY = data_ids[1:]
            dataY.append(self.word2id["<eos>"])
            dataXL = self.SplitListBySeqLength(dataX)
            dataYL = self.SplitListBySeqLength(dataY)
        return (dataXL, dataYL)

def main():
    print('hello')
    myLMDataset = LMDataset(
        vocabPath = './/corpus//charList.txt', \
        dataPath = './/corpus//corpus0_seg_uchar_head2k.txt', \
        seqLength = 16)
    print('olleh')

if __name__ == '__main__':
    main()
