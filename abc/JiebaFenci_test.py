#encoding=utf-8
""" 
Created on 2017-02-17 @author: yongcai
分词
输入(fileName)：需要分词的文件
输出(output_file)：一个txt整合所有文本 每行对应一个文本分词结果
"""

import codecs
import os
import jieba
import jieba.analyse


#Read file and cut
def read_file_cut():

    #create path
    output_file = "./fenci_0003.txt"
    if os.path.exists(output_file):
        os.remove(output_file)
    result = codecs.open(output_file, 'w', 'utf-8')

    fileName = "./0003.txt"
    print(fileName)
    source = open(fileName, 'r')
    line = source.readline()
        
    while line != "":
        line = line.rstrip('\n')
        #line = unicode(line, "utf-8")
        seglist = jieba.cut(line, cut_all=False)  # 精确模式
        output = ' '.join(list(seglist))          # 空格拼接
        #print output
        result.write(output + ' ')               # 空格取代换行'\r\n'
        # result.write("\n")                     #  分词后加一行空
        line = source.readline()
    else:
        result.write('\r\n')
        source.close()

    result.close()


#Run function
if __name__ == '__main__':
    read_file_cut()

