import csv
import math
#import random
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

def load_file(filename):
    with open(filename, 'r') as f:
        lines = csv.reader(f)
        dataset = list(lines)
        #print(dataset[:5])
        dataset = np.asarray(dataset).astype(np.float)
        return dataset

def split_dataset(dataset, splitRatio):
    np.random.shuffle(dataset)
    train_size = (int)(len(dataset)*splitRatio)
    train_data = dataset[:train_size]
    test_data = dataset
    return train_data, test_data

def separateByClass(dataset):
    class_wise_data = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in class_wise_data:
            class_wise_data[vector[-1]] = []
        class_wise_data[vector[-1]].append(vector)
    return class_wise_data

def calculate_mean(numbers):
    return sum(numbers)/len(numbers)

def calculate_sd(numbers):
    average = calculate_mean(numbers)
    #if(len(numbers)-1 != 0):
    variance = sum([pow(number-average, 2) for number in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(datapoints):
    summaries = [(calculate_mean(feature), calculate_sd(feature)) for feature in zip(*datapoints)]
    # because last one would be label hence remove it
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    class_wise_data = separateByClass(dataset)
    summaries = {}
    for class_id, datapoints in class_wise_data.items():
        summaries[class_id] = summarize(datapoints)
    return summaries

def calc_prob(num, avg, sd):
    exponent = math.exp(-(math.pow((num-avg), 2)/math.pow(sd, 2)))
    return (1/(math.sqrt(2*math.pi)*sd))*exponent

def calculateClassProb(summaries, input):
    probabilities = {}
    for class_id, class_summaries in summaries.items():
        probabilities[class_id] = 1
        for i in range(len(class_summaries)):
            avg, sd = class_summaries[i]
            x = input[i]
            probabilities[class_id] *= calc_prob(x, avg, sd)
    return probabilities

def predict(summaries, input):
    probabilities = calculateClassProb(summaries, input)
    bestLabel, bestProb = None, -1
    for class_id, prob in probabilities.items():
        if bestLabel is None or prob > bestProb:
            bestProb = prob
            bestLabel = class_id

    return bestLabel

def getPredictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        predictions.append(predict(summaries, test_set[i]))
    return predictions

def getAccuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0

def usesklearnGNB():
    with open('iris.data','r') as f:
        lines = csv.reader(f)
        dataset = list(lines)
        train_data = []
        label_data = []
        for data in dataset:
            if data:
                train_data.append(np.asarray(data[:4], dtype=float))
                if data[4] == 'Iris-setosa':
                    label_data.append(0)
                elif data[4] == 'Iris-versicolor':
                    label_data.append(1)
                else:
                    label_data.append(2)
        print(len(train_data), len(label_data))

        model = GaussianNB()
        model.fit(train_data, label_data)
        actual_label = label_data
        predicted = model.predict(train_data)

        print(metrics.classification_report(actual_label, predicted))
        print(metrics.confusion_matrix(actual_label, predicted))

def main():
    splitRatio = 0.70
    dataset = load_file('pima-indians-diabetes-data.csv')

    training_set, test_set = split_dataset(dataset, splitRatio)
    print('Split {0} row into training = {1} and test = {2} rows.'.format(len(dataset), len(training_set), len(test_set)))

    print("Preparing Model....")
    summaries = summarizeByClass(training_set)
    print("Model Prepared.")
    print("Testing Model...")
    predictions = getPredictions(summaries, test_set)
    accuracy = getAccuracy(test_set, predictions)

    print('Accuracy : {0}'.format(accuracy))

main()
#usesklearnGNB()
#uncomment above to test in sklearn GNB