# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:44:53 2015

@author: yys
"""

from math import *

def main():
    # Get the data set and features
    dataset, features = createDataSet()
    
    # Build the tree
    tree = treeGrowth(dataset, features)
    
    # Print the tree
    print "[Tree Structure]"
    print tree
    print "[Prediction Result]"
    print predict(tree, {'no surfacing': 1, 'flippers': 1})  
    print predict(tree, {'no surfacing': 1, 'flippers': 0})  
    print predict(tree, {'no surfacing': 0, 'flippers': 1})  
    print predict(tree, {'no surfacing': 0, 'flippers': 0})

def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    features = ['no surfacing', 'flippers']
 
    return dataSet, features
    
def treeGrowth(dataSet,features):
    
    classList = [example[-1] for example in dataSet]
 
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    if len(dataSet[0]) == 1:# no more features
        return classify(classList)  
    
    # Find the best entrophy gain
    bestFeat = findBestSplit(dataSet)#bestFeat is the index of best feature  
    bestFeatLabel = features[bestFeat]
    myTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueFeatValues = set(featValues)
    del (features[bestFeat])
    for values in uniqueFeatValues:
        subDataSet = splitDataSet(dataSet, bestFeat, values)
        myTree[bestFeatLabel][values] = treeGrowth(subDataSet, features)  
    return myTree  

def findBestSplit(dataSet):
    # Get the feature vector
    numFeatures = len(dataSet[0]) - 1 
    best_index = -1
    max_infogain = 0.0
    priori_entropy = calLabelEntrophy(dataSet)
    
    for i in range(numFeatures):
        # calculate the information gain of all features
        feature_values = [example[i] for example in dataSet]
        value_range = set(feature_values)
        # summarize the prob for every unique value of     
        n = len(dataSet)
        i_entrophy = 0
        for v in value_range:
            # filter the observations whose Feature[i] == v
            sub_set = splitDataSet(dataSet, i, v)
            v_count = len(sub_set)
            
            prob_v = v_count/float(n)
            # calculate posterior prob
            i_entrophy += prob_v * calLabelEntrophy(sub_set)
            
            if priori_entropy - i_entrophy > max_infogain:
                max_infogain = priori_entropy - i_entrophy
                best_index = i
    
    return best_index
    
# just two columns, one is a feature and the other is the label
def splitDataSet(dataSet, feature_index, value):
    sub_set = []
    for observation in dataSet:
        if observation[feature_index] == value:
            item = observation[:feature_index]
            item.extend(observation[feature_index+1:])
            sub_set.append(item)
    return sub_set
    
# calculate the shannon entrophy of this data set
def calLabelEntrophy(dataSet):
    n = len(dataSet)
    label_count = {}
    # ergodic all labels
    entrophy = 0
    for i in range(n):
        current_label = dataSet[i][-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
        else:
            label_count[current_label] += 1
    
    for key in label_count:
        prob = label_count[key]/float(n)
        if prob != 0:
            entrophy -= prob * log(prob, 2)
    
    return entrophy

def classify(classList):  
    '''
    find the most common label in the set 
    '''  
    classCount = {}

    for vote in classList:  
        if vote not in classCount.keys():  
            classCount[vote] = 0 
        classCount[vote] += 1
        
    sortedClassCount = [(k,classCount[k]) for k in sorted(classCount.values())] 
    return sortedClassCount[0][0]
    
def predict(tree, new_ob):
    while isinstance(tree, dict):
        key = tree.keys()[0]
        tree = tree[key][new_ob[key]]
    return tree
    
if __name__ == '__main__':
    main()
        
            
        
        
        
            

    

