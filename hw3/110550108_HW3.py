# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y, weight=None):
    if weight is None:
        classes = np.unique(y)
        gini = 0
        for cls in classes:
            p = len(y[np.where(y == cls)]) / len(y)
            gini += p**2
    else:
        weight = np.array(weight)
        total_weight = np.sum(weight)
        classes = np.unique(y)
        gini = 0
        for cls in classes:
            p = np.sum(weight[np.where(y == cls)]) / total_weight
            gini += p**2
    return 1 - gini

# This function computes the entropy of a label array.
def entropy(y, weight=None):
    if weight is None:
        classes = np.unique(y)
        entropy = 0
        for cls in classes:
            p = len((y[np.where(y == cls)])) / len(y)
            entropy += p * np.log2(p)
    else:
        weight = np.array(weight)
        total_weight = np.sum(weight)
        classes = np.unique(y)
        entropy = 0
        for cls in classes:
            p = sum(weight[np.where(y == cls)]) / total_weight
            entropy += p * np.log2(p)
    return -entropy

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, cls=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.cls = cls

# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y, weight=None):
        if self.criterion == 'gini':
            return gini(y, weight)
        elif self.criterion == 'entropy':
            return entropy(y, weight)
    
    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y, weight=None):
        y = y.reshape(-1, 1)
        self.feature_importance = np.zeros(X.shape[1])
        data = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(data, 0, weight)
    
    def build_tree(self, data, depth, weight):
        X, y = data[:, :-1], data[:, -1]
        num_features = X.shape[1]
        if self.max_depth is None or depth < self.max_depth:
            split = self.get_split(data, num_features, weight)
            if split['information_gain'] > 0:
                if weight is None:
                    left_weight, right_weight = None, None
                else:
                    left_weight, right_weight = split['left_weight'], split['right_weight']
                left_subtree = self.build_tree(split['left_data'], depth + 1, left_weight)
                right_subtree = self.build_tree(split['right_data'], depth + 1, right_weight)
                self.feature_importance[split['feature']] += 1
                return Node(split['feature'], split['threshold'], left_subtree, right_subtree)

        y = list(y)
        cls = max(y, key=y.count)
        return Node(cls=cls)
    
    def get_split(self, data, num_features, weight):
        best = {'feature': None, 'threshold': None, 'information_gain': 0,
                'left_data': None, 'right_data': None, 'left_weight': None, 'right_weight': None}
        max_information_gain = 0
        features = np.arange(num_features)
        for feature in features:
            col = data[:, feature]
            thresholds = np.unique(col)
            for threshold in thresholds:
                left_data = data[np.where(col <= threshold)]
                right_data = data[np.where(col > threshold)]
                if len(left_data) and len(right_data):
                    y_total, y_left, y_right = data[:, -1], left_data[:, -1], right_data[:, -1]
                    if weight is None:
                        left_weight, right_weight = None, None
                        left_ratio, right_ratio = len(y_left) / len(y_total), len(y_right) / len(y_total)
                    else:
                        left_weight = weight[np.where(col <= threshold)]
                        right_weight = weight[np.where(col > threshold)]
                        left_ratio = np.sum(left_weight) / np.sum(weight)
                        right_ratio = np.sum(right_weight) / np.sum(weight)
                    information_gain = self.impurity(y_total, weight) - (left_ratio * self.impurity(y_left, left_weight) + right_ratio * self.impurity(y_right, right_weight))
                    if information_gain > max_information_gain:
                        max_information_gain = information_gain
                        best = {'feature': feature, 'threshold': threshold, 'information_gain': information_gain,
                                'left_data': left_data, 'right_data': right_data,
                                'left_weight': left_weight, 'right_weight': right_weight}
        return best
    
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        pred_y = []
        for x in X:
            predicted = self.make_predict(x, self.root)
            pred_y.append(predicted)
        return np.array(pred_y)
    
    def make_predict(self, x, node):
        if node.cls != None:
            return node.cls

        if x[node.feature] <= node.threshold:
            return self.make_predict(x, node.left)
        else:
            return self.make_predict(x, node.right)
    
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        plt.title("Feature Importance")
        plt.barh(columns, self.feature_importance)
        plt.show()

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.clfs = []
        self.alpha = []

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        n_samples = X.shape[0]
        weight = np.full(n_samples, 1 / n_samples)
        for _ in range(self.n_estimators):
            clf = DecisionTree(max_depth=1, criterion=self.criterion)
            clf.fit(X, y, weight)
            y_pred = clf.predict(X)
            self.clfs.append(clf)

            error_samples = (y != y_pred)
            error = np.sum(error_samples * weight)
            alpha = 0.3 * np.log((1 - error) / error)
            self.alpha.append(alpha)
            weight *= np.exp([alpha if error else - alpha for error in error_samples])
            weight /= np.sum(weight)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        y_pred = []
        for x in X:
            total = 0
            for i, clf in enumerate(self.clfs):
                prediction = clf.make_predict(x, clf.root)
                if prediction == 0:
                    total -= self.alpha[i]
                else:
                    total += self.alpha[i]
            if total > 0:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return np.array(y_pred)

# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))
    
# Plotting the feature importance
    # columns = train_df.iloc[:, :-1].columns
    # tree.plot_feature_importance_img(columns)

# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='entropy', n_estimators=7)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
