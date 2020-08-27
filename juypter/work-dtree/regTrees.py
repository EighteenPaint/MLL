'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
import numpy as np

inf = float('inf')


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # map all elements to float()
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    """
    return 前大后小
    这里代码有问题
    # mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :][0]  # 这里代码有问题，下方是修改后的代码
    # mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :][0]
    # return mat0, mat1

    修改后应为：
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]  #
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    """

    mat0 = []
    mat1 = []
    lineIndex = np.nonzero(dataSet[:, feature] > value)[0]
    if len(lineIndex) != 0:
        mat0 = dataSet[lineIndex, :]  # ???
    lineIndex = np.nonzero(dataSet[:, feature] <= value)[0]
    if len(lineIndex) != 0:
        mat1 = dataSet[lineIndex, :]
    return mat0, mat1


def regLeaf(dataSet):  # returns the value used for each leaf
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def linearSolve(dataSet):  # helper function used in two places
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))  # create a copy of data with 1 in 0th postion
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]  # and strip out Y
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):  # create linear model and return coeficients
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y - yHat, 2))


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """

    :param dataSet:
    :param leafType:
    :param errType:
    :param ops: 可以调节的参数，会影响到回归树的构建
    :return:
    """
    tolS = ops[0]
    tolN = ops[1]
    # if all the target variables are the same value: quit and return value
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # 最后一列就是输出值，如果最后一行都一样了，就不需要进行划分
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    # the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(*dataSet[:, featIndex].flatten().tolist()):  # 使用*进行解包，也可以使用[0]进行取值，前提是先用flatten压平
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  # 至少要有一条数据
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:  # 下降必须达到某个数值才算真正的下降
        return None, leafType(dataSet)  # exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  # exit cond 3
        return None, leafType(dataSet)
    return bestIndex, bestValue  # returns the best feature to split on
    # and the value used for that split


def createTree(dataSet, leafType=regLeaf, errType=regErr,
               ops=(1, 4)):  # assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)  # choose the best split
    if feat == None: return val  # if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    rSet, lSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)  # if we have no test data collapse the tree
    if isTree(tree['right']) or isTree(tree['left']):  # if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                       np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat


def predictByRegTree(inputTree, testVec):
    """
    {
	'spInd': 22,
	'spVal': 105.9,
	'left': {
		'spInd': 22,
		'spVal': 117.2,
		'left': 0.006944444444444444,
		'right': {
			'spInd': 1,
			'spVal': 19.34,
			'left': 0.07692307692307693,
			'right': {
				'spInd': 0,
				'spVal': 14.34,
				'left': 1.0,
				'right': 0.25
			}
		}
	},
	'right': {
		'spInd': 24,
		'spVal': 0.1733,
		'left': 0.0,
		'right': {
			'spInd': 27,
			'spVal': 0.1571,
			'left': 0.4,
			'right': 0.9803921568627451
		}
	}
}
    :param tree:
    :param test:
    :return:
    """

    # firstStr = list(inputTree.keys())[0]  # 树根
    # secondDict = inputTree[firstStr]  # 子树
    # featIndex = featLabels.index(firstStr)  # 找到树根标签所对应的属性索引值
    # for key in secondDict.keys():
    #     if testVec[featIndex] == key:
    #         if type(secondDict[key]).__name__ == "dict":  # 如果还有子树，说明继续进行决策
    #             classLabel = classify(secondDict, featLabels, testVec)  # 这里进行了递归
    #         else:
    #             classLabel = secondDict[key]  # 没有子树的话，说明已经搜索完毕
    # return classLabel
    index = inputTree['spInd']
    value = inputTree['spVal']
    if testVec[index] > value:
        if isinstance(inputTree['right'], dict):
            return predictByRegTree(inputTree['right'], testVec)
        else:
            return inputTree['right']
    else:
        if isinstance(inputTree['left'], dict):
            return predictByRegTree(inputTree['left'], testVec)
        else:
            return inputTree['left']


def preBreast(regTree, testVecs, real):
    m, n = np.shape(testVecs)
    result = []
    for testVec in testVecs:
        ret = predictByRegTree(regTree, testVec)
        if (ret >= 0.5):
            result.append(1)
        else:
            result.append(0)
    # print(result)
    # print(real)
    correct = result ^ real
    # print(correct)
    accuracy = np.sum(correct) / len(real)
    # print("正确率：",1 - accuracy)
    return 1 - accuracy


if __name__ == '__main__':
    # myDat = loadDataSet("ex0.txt")
    # myDat = np.mat(myDat)
    # tree = createTree(myDat)
    # result = predictByRegTree(tree, [1.000000, 0.182603])
    # print("predict value", result)
    # print("real value", 0.063908)
    print(sigmod(1), sigmod(0.2), sigmod(0.8))
