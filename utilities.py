# -- coding:utf-8

def ExtractWordListFrmSegFile(segCorpusIn, wordListOut):
    with open(segCorpusIn, 'r') as fIn:
        with open(wordListOut, 'w') as fOut:
            wordset = set()
            for line in fIn.readlines():
                items = line.strip().split()
                wordset.update(items)
            wordlist = list(wordset)
            wordlist.sort()
            output_line = '\n'.join(wordlist)
            fOut.write(output_line)
    pass

def main():
    print('hello')
    ExtractWordListFrmSegFile(
        segCorpusIn = './/corpus//corpus0_seg_char.txt', \
        wordListOut = './/corpus//charList.txt')
    print('olleh')

if __name__ == '__main__':
    main()
