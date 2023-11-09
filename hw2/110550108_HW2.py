# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = np.zeros(6)
        self.intercept = -1.4
        self.losses = []

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        for i in range(self.iteration):
            z = X.dot(self.weights) + self.intercept
            y_pred = self.sigmoid(z)
            loss = self.Cross_Entropy_Error(y, y_pred)
            self.losses.append(loss)
            
            weight_gradient = (X.T.dot(y_pred - y)) / len(y)
            intercept_gradient = (y_pred - y).mean()
            self.weights -= weight_gradient * self.learning_rate
            self.intercept -= intercept_gradient * self.learning_rate
            
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        z = X.dot(self.weights) + self.intercept
        y_pred = self.sigmoid(z)
        y_pred = (y_pred > 0.5).astype(int)
        return y_pred

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def Cross_Entropy_Error(self, y_real, y_pred):
        cost = -y_real * np.log(y_pred) - (1 - y_real) * np.log(1 - y_pred) 
        cost = np.sum(cost)
        return cost

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = np.zeros((2, 2))
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        class0 = X[np.where(y == 0)]
        self.m0 = class0.mean(axis=0)
        class1 = X[np.where(y != 0)]
        self.m1 = class1.mean(axis=0)
        
        for v0 in class0:
            s0 = v0 - self.m0
            self.sw += np.outer(s0, s0)
        for v1 in class1:
            s1 = v1 - self.m1
            self.sw += np.outer(s1, s1)

        s = self.m1 - self.m0
        self.sb = np.outer(s, s)
        
        inverse_sw = np.linalg.inv(self.sw)
        self.w = s @ inverse_sw
        self.w /= np.linalg.norm(self.w)

    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        
        projected_X = X @ self.w
        projected_m0 = self.m0 @ self.w
        projected_m1 = self.m1 @ self.w

        for i in range(X.shape[0]):
            dist_to_m0 = np.abs(projected_X[i] - projected_m0)
            dist_to_m1 = np.abs(projected_X[i] - projected_m1)

            if dist_to_m0 < dist_to_m1:
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        return y_pred

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X, y_pred):
        slope = self.w[1] / self.w[0]
        intercept = 0
        x = np.linspace(-50, 75)
        y = slope * x + intercept
        
        class0, class1 = X[np.where(y_pred == 0)], X[np.where(y_pred != 0)]
        projected_class0_x, projected_class0_y = self.project(class0)[:, 0], self.project(class0)[:, 1]
        projected_class1_x, projected_class1_y = self.project(class1)[:, 0], self.project(class1)[:, 1]
        
        fig, ax = plt.subplots()
        ax.set_title(f'Projection Line: w={slope}, b={intercept}')
        ax.set_ylim(75, 190)
        ax.plot(x, y)
        ax.plot([class0[:, 0], projected_class0_x], [class0[:, 1], projected_class0_y], c='slategrey', alpha=0.05)
        ax.plot([class1[:, 0], projected_class1_x], [class1[:, 1], projected_class1_y], c='slategrey', alpha=0.05)
        ax.scatter(class0[:, 0], class0[:, 1], c='r', s=10)
        ax.scatter(class1[:, 0], class1[:, 1], c='b', s=10)
        ax.scatter(projected_class0_x, projected_class0_y, c='r', s=10)
        ax.scatter(projected_class1_x, projected_class1_y, c='b', s=10)
        plt.show()
        
    def project(self, x):
        projection = np.outer((x @ self.w), self.w)
        return projection

# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    
# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    
    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.00032, iteration=225000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    
    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"
