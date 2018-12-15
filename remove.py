# encoding:utf-8
import sys
from imp import reload
reload(sys)

import re
import jieba
import collections


class Remove:
      # numpy 转化为 list
    def ndarrayToList(self, dataArr):
        dataList = []
        m, n = dataArr.shape
        for i in range(m):
            for j in range(n):
                dataList.append(dataArr[i, j])
        return dataList
    # 去掉字符串、特殊符号

    def removeStr(self, listData):
        strData = "".join(listData)
        removeStrData = re.sub(
            "[\s+\!\,$^*()+\"\']+:|[+——！，,《》“”〔【】；：。？、�./-~@#￥……&*（）]+", "", strData)
        return removeStrData

    # 创建停用词列表
    def stopwordslist(self, filePath):
        stopword = [line.strip() for line in open(
            filePath, 'r', encoding='utf-8').readlines()]
        return stopword
    # 对数据集分词、去停用词

    def wordSplit(self, data):
        stopwords = self.stopwordslist('stopwords.txt')  # 创建通用词列表
        word = self.ndarrayToList(data)
        m = len(word)
        wordList = []
        for i in range(m):
            if isinstance(word[i], collections.Iterable):
                rowListRemoveStr = self.removeStr(word[i])    # 去特殊符号
            else:
                rowListRemoveStr = ""

            rowList = [eachWord for eachWord in jieba.cut(
                rowListRemoveStr)]  # 分词
            removeStopwordList = []
            for eachword in rowList:
                if eachword not in stopwords and eachword != '\t' and eachword != ' ':
                    removeStopwordList.append(eachword)
            wordList.append(removeStopwordList)
        return wordList
