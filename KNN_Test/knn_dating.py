#使用KNN算法改进约会网站的配对效果
#K-近邻算法：已知一系列带标签的数据集，通过计算未知样本与数据集中样本的欧式距离，
# 并对距离进行排序，取距离最近的K个样本的标签，将未知样本归到距离最近的K个样本相同的标签中
import numpy as np
import operator
#K-近邻算法
def classify0(inX,dataSet,Labels,k):#inX:1*3; dataSet:1000*3; Labels:1000*1
    dataSetSize=np.shape(dataSet)[0]#样本数据集个数1000
    diffMat=np.tile(inX,(dataSetSize,1))-dataSetSize#因为inX的大小与dataSetSize不一致，所以要进行数组的赋值（延y轴）,将1*3延y轴为1000*3
    sqDiffMat=diffMat**2#对每个array数组元素**2
    sqDistance=sqDiffMat.sum(axis=1)#按行求和
    sqDistance=sqDistance**0.5
    sqDistance=sqDistance.argsort()
    classCount={}
    for i in range(k):
        votelLabel=Labels[sqDistance[i]]
        classCount[votelLabel]=classCount.get(votelLabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
#将文本记录转化为numpy数组
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()#返回列表
    numbersOfLines=len(arrayOLines)#文本文件行数
    returnMat=np.zeros((numbersOfLines,3))#3个特征
    classLabelVector=[]#标签集
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        if listFromLine[-1]=='largeDoses':
            classLabelVector.append(1)
        elif listFromLine[-1]=='smallDoses':
            classLabelVector.append(0)
        else:
            classLabelVector.append(-1)
        index+=1
    return returnMat,classLabelVector
#归一化特征值:在计算欧式距离的时候，数字差值最大的属性对计算结果的影响最大，但是三个特征的权重是一样的，所以要归一化
def autoNorm(dataSet):#1000*3
    minValue=dataSet.min(0)#1*3
    maxValue = dataSet.max(0)#1*3
    ranges=maxValue-minValue#1*3
    normDataSet=np.zeros(np.shape(dataSet))
    m=np.shape(dataSet)[0]#1000
    normDataSet=dataSet-np.tile(minValue,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1))#对应位置，对应相除，1*3->1000*3
    return normDataSet,ranges,minValue
#测试
def datingClassTest():
    hoRatio=0.1#选择10%的数据作为测试数据
    datingDataMax,datingDataLabels=file2matrix('E:\AI_Test\knn+dating\datingTestSet.txt')
    normDataSet,ranges,minValue=autoNorm(datingDataMax)
    m=int(np.shape(datingDataMax)[0])#数据集的大小
    numberOfTest=int(hoRatio*m)#测试集数目
    numberOfTrain=int(-numberOfTest)#训练集数量
    errorCount=0
    for i in range(numberOfTest):
        classifyResult=classify0(datingDataMax[i,:],datingDataMax[numberOfTest:m,:],datingDataLabels[numberOfTest:m],3)
        if classifyResult!=datingDataLabels[i]:
            errorCount+=1
        print("分类器分类标签: %d,真实标签 %d"%(classifyResult,datingDataLabels[i]))
    print("分类错误率： %f"%(errorCount/numberOfTest))
if __name__ == '__main__':
    datingClassTest()























