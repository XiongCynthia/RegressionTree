import torch
from sklearn.metrics import mean_squared_error as mse

class RegressionTree:
    '''
    A class for fitting and predicting data on a decision tree for regression.
    '''        
    def __init__(self, min_samples_split=20, max_depth=5):
        '''
        Args:
            min_samples_split: The minimum number of observations in a sample that can be split
            max_depth: The maximum depth of the tree
        '''
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        return

    
    class Node:
        '''
        A node of a decision tree.
        '''
        def __init__(self):
            self.left = None # Left node
            self.right = None # Right node
            self.depth = None
            self.val = None # Upper bound if non-leaf, or predictive value if leaf
            self.feature_idx = None # Column index of a feature if non-leaf, or None if leaf
            return
        
        
    def fit(self, X:torch.Tensor, y:torch.Tensor):
        '''
        Fits data on a regression tree.

        Args:
            X: Training data
            y: Target values
        '''
        # Number of leaves
        # This serves no actual purpose besides counting leaves
        self.leaves_ = 0

        # Initialize root
        self.root_ = self.Node()
        self.root_.depth = 1

        # Determine decision for the root node
        bound, col_idx = self.__find_best_tree(X, y)

        # Store upper bound and feature indxe of the tree with the lowest mse in the root node
        self.root_.val = bound
        self.root_.feature_idx = col_idx
        
        self.__build_tree(X, y)
        return

    
    def predict(self, X:torch.Tensor):
        '''
        Predicts using the fitted regression tree.

        Args:
            X: Sample data
        '''
        if not self.is_fitted():
            raise Exception("This RegressionTree instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        if len(X.shape) == 0: # If X is a scalar
            return torch.tensor(self.__search_tree(X))

        y_preds = []
        for obs in X:
            y_preds.append(self.__search_tree(obs))
        return torch.Tensor(y_preds)
    

    def is_fitted(self):
        '''
        Indicates whether the fit() method has been called.
        '''
        if hasattr(self, 'root_'):
            return True
        else:
            return False
    

    def __build_tree(self, X, y):
        '''
        Begins recursively building the regression tree, starting from the root.

        Args:
            X: Training data
            y: Target values
            root: Root node
        '''
        # Split rows of the training data by the determined upper bound
        if len(X.shape) == 1:
            rows_idx = X < self.root_.val
        else:
            rows_idx = X[:,self.root_.feature_idx] < self.root_.val

        self.root_.left = self.__recurs_build_tree(X[rows_idx], y[rows_idx], self.root_)
        self.root_.right = self.__recurs_build_tree(X[~rows_idx], y[~rows_idx], self.root_)
        return


    def __recurs_build_tree(self, X, y, prev:Node):
        '''
        Recursively builds the regression tree.

        Args:
            X: Training data
            y: Target values
            prev: Previous node
        '''
        is_leaf = False # Should this current node be a leaf?
        if self.max_depth != None:
            if prev.depth+1 == self.max_depth:
                is_leaf = True
        if X.shape[0] < self.min_samples_split:
            is_leaf = True
        
        node = self.Node() # Current node
        node.depth = prev.depth + 1
        if is_leaf: # Base case
            node.val = torch.mean(y)
            self.leaves_ += 1
            return node
        else: # Recursive case
            bound, col_idx = self.__find_best_tree(X, y)
            node.val = bound
            node.feature_idx = col_idx

            # Split rows of the training data by the determined upper bound
            if len(X.shape) == 1:
                rows_idx = X < bound
            else:
                rows_idx = X[:,col_idx] < bound

            node.left = self.__recurs_build_tree(X[rows_idx], y[rows_idx], node)
            node.right = self.__recurs_build_tree(X[~rows_idx], y[~rows_idx], node)

            return node
        
    
    def __find_best_tree(self, X:torch.Tensor, y:torch.Tensor):
        '''
        Calculates the best possible tree when the bound splits the data into two.

        Args:
            X: Training data
            y: Target values 

        Returns:
            tuple[torch.Tensor, None]: If X is a 1-D tensor, return the upper bound.
            tuple[torch.Tensor, int]: If X has multiple dimensions, return the upper bound
            and its associated error.  
        '''
        # If there is only 1 feature, simply return the bound
        if len(X.shape) == 1:
            bound, error = self.__find_best_bound(X, y)
            return bound, None
        
        # Get trees with the lowest mean squared errors for each feature in training data
        bounds = []
        errors = []
        for col_idx in range(X.shape[1]):
            bound, error = self.__find_best_bound(X[:,col_idx], y)
            bounds.append(bound)
            errors.append(error)
        # Find and return upper bound and mse of best tree (i.e., tree with lowest mse)
        min_idx = errors.index(min(errors))
        return bounds[min_idx], min_idx
    

    def __find_best_bound(self, Xn:torch.Tensor, y:torch.Tensor):
        '''
        Calculates the best bounds for splitting the data into two, and returns
        the upper bound and its resulting mean squared error.

        Args:
            Xn: A column (or feature) from the training data
            y: Target values
        '''
        # If there are only 2 (or 1) observations, simply let the bound split them in half
        if Xn.shape[0] <= 2:
            y_pred = torch.mean(y)
            return y_pred, sum(y - [y_pred]*Xn.shape[0])
        errors = []
        for val in Xn:
            idx = Xn < val
            if (len(Xn[idx]) == 0) or (len(Xn[~idx]) == 0):
                # This will occur if val is the minimum or maximum value in Xn
                errors.append(-1) # Use a -1 placeholder to represent an invalid upper bound for splitting
                continue
            y_pred = torch.mean(y[idx])
            error = mse(y[idx], [y_pred]*torch.sum(idx))

            y_pred = torch.mean(y[~idx])
            error += mse(y[~idx], [y_pred]*torch.sum(~idx))

            errors.append(error)
        # Get index of minimum mse (while ignoring -1 values)
        min_idx = errors.index(min([val for val in errors if val != -1]))
        return Xn[min_idx], errors[min_idx]
    
    
    def __search_tree(self, feature_vals):
        '''
        Begins recursively searching through the decision tree to return a prediction 
        using the given values.

        Args:
            feature_vals: Values for each feature in X
        '''
        return self.__recurs_search_tree(feature_vals, self.root_)
        

    def __recurs_search_tree(self, feature_vals:torch.Tensor, node:Node):
        '''
        Searches through the decision tree recursively.

        Args:
            feature_vals: Values for each feature in X
            node: Current node
        '''
        if node.feature_idx == None: # Base case
            # Since this node is a leaf, return its value
            return node.val
        else: # Recursive case
            # Go left or right?
            if len(feature_vals.shape) == 0: # If there is only 1 feature, and its a scalar
                go_left = feature_vals < node.val
            else:
                go_left = feature_vals[node.feature_idx] < node.val

            if go_left:
                return self.__recurs_search_tree(feature_vals, node.left)
            else:
                return self.__recurs_search_tree(feature_vals, node.right)
