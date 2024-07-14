import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection

"""
- -- binary classification
- -- multiclass classification
- -- multilabel classification
- -- single column classfication
- -- multi column classification
- -- holdout
"""

class CrossValidation:
    def __init__(
            self,
            df,
            target_cols,
            problem_type = 'binary_classification',
            num_folds = 5,
            shuffle = True,
            random_state = 42
        ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle  = shuffle
        self.random_state = random_state


        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        self.dataframe["kfold"] = -1
    
    def split(self):
        if self.problem_type in ('binary_classification', 'multiclass_classification'):
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value in target column")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits = self.num_folds,
                                                     shuffle = False)
                for fold, (train_idx, val_idx) in enumerate(kf.split(X = self.dataframe, y = self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold
        elif self.self.problem_type in ('single_col_regression', 'multi_col_regression'):
            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            if self.num_targets < 2 and self.problem_type == "multi_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X = self.dataframe)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        else:
            raise Exception("Invalid problem type")
        
        return self.dataframe
            
            

if __name__ == "__main__":
    print("..program started..")
    df = pd.read_csv("input/train.csv")
    cv = CrossValidation(df,
                          shuffle=True, 
                          target_cols=["Exited"], 
                          num_folds=5,
                          problem_type="binary_classification")
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())

