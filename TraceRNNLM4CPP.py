# -- coding:utf-8

from __future__ import division

import argparse
import torch
import numpy as np

from RNNLM import RNNConfig, RNNLM

def TraceRNNLM():
    parser = argparse.ArgumentParser(description='Trace RNNLM for CPP deployment')
    parser.add_argument('--configPath', type=str, default='',
                        help='the file path for model config')
    parser.add_argument('--modelPath', type=str, default='',
                        help='the file path of LM model')
    parser.add_argument('--batchSize', type=int, default=32,
                        help='the batch size')
    parser.add_argument('--savePath', type=str,  default='',
                        help='the path to save the final model')
    args, _ = parser.parse_known_args()
    theRNNConfig = RNNConfig()
    theRNNConfig.readfrmjson(args.configPath)
    model = RNNLM(theRNNConfig)
    model.load_state_dict(torch.load(args.modelPath))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    hidden = model.init_hidden(args.batchSize).to(device)
    input_np = [i for i in range(0,10)]
    input = torch.from_numpy(np.array(input_np, dtype=np.int64)).to(device)
    input_ex = torch.unsqueeze(input,1)
    model_trace = torch.jit.trace(model, (input_ex, hidden) )
    model_trace.save(args.savePath)
    pass

def main():
    print('hello')
    TraceRNNLM()
    print('olleh')

if __name__ == '__main__':
    main()

