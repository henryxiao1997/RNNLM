# -- coding:utf-8

from __future__ import division

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import numpy as np

from LMData import LMDataset
from RNNLM import RNNConfig, RNNLM

CUDA_LAUNCH_BLOCKING="1"

def TrainRNNLM():
    parser = argparse.ArgumentParser(description='Train RNNLM')
    parser.add_argument('--vocabPath', type=str, default='',
                        help='the file path for vocabulary')
    parser.add_argument('--dataPath', type=str, default='',
                        help='the file path for training data')
    parser.add_argument('--configPath', type=str, default='',
                        help='the file path for model config')
    parser.add_argument('--seqLength', type=int, default=16,
                        help='the sequence length for one sample')
    parser.add_argument('--batchSize', type=int, default=32,
                        help='the batch size')
    parser.add_argument('--epochNum', type=int, default=10,
                        help='the number of epochs')
    parser.add_argument('--learningRate', type=float, default=1e-4,
                        help='the initial learning rate')
    parser.add_argument('--savePath', type=str,  default='',
                        help='the path to save the final model')
    args, _ = parser.parse_known_args()

    theRNNConfig = RNNConfig()
    theRNNConfig.readfrmjson(args.configPath)
    model = RNNLM(theRNNConfig)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learningRate)
    hidden = model.init_hidden(theRNNConfig, args.batchSize)
    savehidden = copy.deepcopy(hidden)
    print("--------------------------- Training ---------------------------")
    tBegin= time.time()
    for epoch in tqdm(range(args.epochNum)):
        theLMDataset = LMDataset(vocabPath=args.vocabPath, dataPath=args.dataPath, seqLength=16)
        #theDataLoader = DataLoader(theLMDataset, batch_size=args.batchSize, shuffle=False, drop_last=True, num_workers=4)
        theDataLoader = DataLoader(theLMDataset, batch_size=args.batchSize, shuffle=False, drop_last=True) 
        totalBsNum = theLMDataset.__len__() // args.batchSize
        costsEpoch = 0.0
        for i, batch_data in enumerate(theDataLoader):
            tBeginInBs = time.time()
            inputs = batch_data[0].transpose(0, 1).contiguous().to(device)
            targets = batch_data[1].transpose(0, 1).contiguous().to(device)
            model.zero_grad()
            hidden = copy.deepcopy(savehidden)
            outputs, hidden = model(theRNNConfig, inputs, hidden)
            tt = torch.squeeze(targets.view(-1, args.batchSize * args.seqLength))
            loss = criterion(outputs.view(-1, theRNNConfig.vocab_size), tt)
            cost = torch.Tensor.item(loss.data)
            costsEpoch += cost
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()
            if i % 10 == 0:
                print("the perplexity of {} steps ({}%): {}  speed: {}"\
                    .format(i, (i*100/totalBsNum),np.exp(cost), args.batchSize*args.seqLength/(time.time()-tBeginInBs)) )
        print("the perplexity of {} epoches: {} ".format(epoch, np.exp(costsEpoch/(i+1))) )
    tEnd = time.time()
    print("the total time consuming: {}".format(tEnd-tBegin))
    print("----------------------- finish Training -------------------------")
    with open(args.savePath, 'wb') as fOut:
        torch.save(model.state_dict(), fOut)
    print("------------------------ dumped model ---------------------------")
    pass

def main():
    print('hello')
    TrainRNNLM()
    print('olleh')

if __name__ == '__main__':
    main()

