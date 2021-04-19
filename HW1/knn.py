import numpy as np
import statistics

COLUMNS = 86
ROWS = 8000
NEIGHBORS = 1
TEST_FILE_NAME = 'test_pub.csv'
TRAIN_FILE_NAME = 'train.csv'

def importData(file_name):
    data = np.genfromtxt(file_name, delimiter=',')

    #Deletes the first row of column names and returns the np matrix
    return np.delete(data, 0, 0)

#x and y are both np arrays of features, returns the euclidian distance
def getNeighbors(training_data, test_point, numNeighbors):
    dists = np.linalg.norm(training_data - test_point, axis=1)
    return np.argsort(dists)[:NEIGHBORS]

def kNNClassify(train_data, test_point, numNeighbors):
    #Doesn't use the first and last column since it's the income and id features
    nearestK = getNeighbors(train_data[:, 1:COLUMNS], test_point[1:COLUMNS], NEIGHBORS)
    return int(statistics.mode([train_data[i, COLUMNS] for i in nearestK]))

def evaluate_performance(test_data, train_data):
    correctClass = 0
    totalTested = 0
    for i in test_data:
        output = kNNClassify(train_data, i, NEIGHBORS)
        if output == int(i[COLUMNS]):
            correctClass += 1
        totalTested += 1
        print(int(i[0]), output)
    print("performance:", correctClass / totalTested)

if __name__ == "__main__":
    train_data = importData(TRAIN_FILE_NAME)
    #test_data = importData('test_pub.csv')

    test_data = train_data[(int(3*ROWS/4) + 1):]
    train_data = train_data[:int(3*ROWS/4)]

    print("id, income")
    correctClass = 0
    totalTested = 0
    for i in test_data:
        output = kNNClassify(train_data, i, NEIGHBORS)
        if output == int(i[COLUMNS]):
            correctClass += 1
        totalTested += 1
        print(int(i[0]), output)
    print("performance:", correctClass / totalTested)


