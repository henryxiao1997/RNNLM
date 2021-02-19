nohup python --vocabPath, ".//corpus//charList.txt", \
    --dataPath, ".//corpus//corpus0_seg_uchar.txt", \
    --configPath, ".//configs//config_lstm_128dim_2L.json", \
    --seqLength, 16, \
    --batchSize, 512, \
    --epochNum, 10,  \
    --learningRate, 1e-4, \
    --savePath, ".//models/model.pt" \
	> run_log.txt &2 > 1
