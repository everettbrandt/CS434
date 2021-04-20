import numpy as np
import statistics

COLUMNS = 86
ROWS = 8000
TEST_FILE_NAME = 'test_pub.csv'
TRAIN_FILE_NAME = 'train.csv'

def importData(file_name):
    data = np.genfromtxt(file_name, delimiter=',')

    #Deletes the first row of column names and returns the np matrix
    return np.delete(data, 0, 0)

#x and y are both np arrays of features, returns the euclidian distance
def getNeighbors(training_data, test_point, numNeighbors):
    dists = np.linalg.norm(training_data - test_point, axis=1)
    return np.argsort(dists)[:numNeighbors]

def kNNClassify(train_data, test_point, numNeighbors):
    #Doesn't use the first and last column since it's the income and id features
    nearestK = getNeighbors(train_data[:, 1:COLUMNS-1], test_point[1:COLUMNS-1], numNeighbors)

    #Returns the class that receives the most votes
    return int(statistics.mode([train_data[i, COLUMNS] for i in nearestK]))

def evaluate_performance(test_data, train_data, numNeighbors):
    correctClass = 0
    totalTested = 0
    for i in test_data:
        output = kNNClassify(train_data, i, numNeighbors)
        if output == int(i[COLUMNS]):
            correctClass += 1
        totalTested += 1
    return correctClass / totalTested

def classify(train_data, test_data, numNeighbors):
    print("id, income")
    for i in test_data:
        print(int(i[0]), int(kNNClassify(train_data, i, numNeighbors)))

def fourFoldPerf(data, numNeighbors):
    performances = list()
    #evaluates the performance using the first 6000 rows as training and rows 6001 - 8000 as testing
    train_data = data[:int(3*ROWS/4)]
    test_data = data[(int(3*ROWS/4) + 1):]
    performances.append(evaluate_performance(train_data, test_data, numNeighbors))

    #evaluates rows 0-4000 and 6,001 - 8,000 as training, 4001 - 6000 as testing
    train_data = data[:int(ROWS/2)]
    train_data = np.concatenate((train_data, data[int(3*ROWS/4)+1:ROWS]), axis = 0)
    test_data = data[int(ROWS/2 + 1):int(3*ROWS/4)]
    performances.append(evaluate_performance(train_data, test_data, numNeighbors))

    #evaluates rows 0-2000 and 4,001 - 8,000 as training, 4001 - 6000 as testing
    train_data = data[:int(ROWS/4)]
    train_data = np.concatenate((train_data, data[int(ROWS/2)+1:ROWS]), axis = 0)
    test_data = data[int(ROWS/2) + 1:int(3*ROWS/4)]
    performances.append(evaluate_performance(train_data, test_data, numNeighbors))

    #evaluates rows 2001-8000 as training, 0 - 2000 as testing
    train_data = data[(int(1*ROWS/4) + 1):] 
    test_data = data[:int(1*ROWS/4)]
    performances.append(evaluate_performance(train_data, test_data, numNeighbors))
    return performances
    
if __name__ == "__main__":
    train_data = importData(TRAIN_FILE_NAME)
    test_data = importData('test_pub.csv')
    
    classify(train_data, test_data, 9)

    """print("neighbors, performance")
    neighbors = 1
    print(neighbors, evaluate_performance(train_data, train_data, neighbors))
    neighbors = 3
    print(neighbors, evaluate_performance(train_data, train_data, neighbors))
    neighbors = 5
    print(neighbors, evaluate_performance(train_data, train_data, neighbors))
    neighbors = 7
    print(neighbors, evaluate_performance(train_data, train_data, neighbors))
    neighbors = 9
    print(neighbors, evaluate_performance(train_data, train_data, neighbors))
    neighbors = 99
    print(neighbors, evaluate_performance(train_data, train_data, neighbors))
    neighbors = 999
    print(neighbors, evaluate_performance(train_data, train_data, neighbors))
    neighbors = 8000
    print(neighbors, evaluate_performance(train_data, train_data, neighbors))"""

    """print("neighbors, mean performance, variance")
    neighbors = 1
    performance = fourFoldPerf(train_data, neighbors)
    print(neighbors, statistics.mean(performance), statistics.variance(performance))
    neighbors = 3
    performance = fourFoldPerf(train_data, neighbors)
    print(neighbors, statistics.mean(performance), statistics.variance(performance))
    neighbors = 5
    performance = fourFoldPerf(train_data, neighbors)
    print(neighbors, statistics.mean(performance), statistics.variance(performance))
    neighbors = 7
    performance = fourFoldPerf(train_data, neighbors)
    print(neighbors, statistics.mean(performance), statistics.variance(performance))
    neighbors = 9
    performance = fourFoldPerf(train_data, neighbors)
    print(neighbors, statistics.mean(performance), statistics.variance(performance))
    neighbors = 99
    performance = fourFoldPerf(train_data, neighbors)
    print(neighbors, statistics.mean(performance), statistics.variance(performance))
    neighbors = 999
    performance = fourFoldPerf(train_data, neighbors)
    print(neighbors, statistics.mean(performance), statistics.variance(performance))
    neighbors = 8000
    performance = fourFoldPerf(train_data, neighbors)
    print(neighbors, statistics.mean(performance), statistics.variance(performance))"""


