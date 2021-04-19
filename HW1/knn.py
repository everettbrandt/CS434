import numpy as np

COLUMNS = 87
ROWS = 8000
NEIGHBORS = 1

def importData(file_name):
    data = np.genfromtxt(file_name, delimiter=',')

    #Deletes the first row of column names and returns the np matrix
    return np.delete(data, 0, 0)

#x and y are both np arrays of features, returns the euclidian distance
def getNeighbors(training_data, test_point, numNeighbors):
    dists = np.linalg.norm(training_data - test_point, axis=1)
    print(dists)
    print(np.argsort(dists))

def kNNClassify(X, Y, x, k):
    print()

if __name__ == "__main__":
    train_data = importData('train.csv')
    test_data = importData('test_pub.csv')

    #Calculates distance between vectors, doesn't use id or income as a feature

    getNeighbors(train_data[:, 1:COLUMNS - 1], train_data[1, 1: COLUMNS - 1], NEIGHBORS)
    #calculate_distance(train_data[0, 1:COLUMNS - 1], train_data[1, 1:COLUMNS - 1])
