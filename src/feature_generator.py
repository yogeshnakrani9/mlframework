import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from typing import List, Union, Dict, Optional

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()

    def handle_missing_values(self, strategy: Dict[str, str]):
        """
        Handle missing values in the DataFrame.
        
        :param strategy: Dict with column names as keys and imputation strategy as values.
                         Strategies: 'mean', 'median', 'most_frequent', 'constant'
        """
        for col, method in strategy.items():
            if method == 'constant':
                self.df[col].fillna(-999999, inplace=True)
            else:
                imputer = SimpleImputer(strategy=method)
                self.df[col] = imputer.fit_transform(self.df[[col]])
        return self

    def encode_categorical(self, columns: Dict[str, str]):
        """
        Encode categorical variables.
        
        :param columns: Dict with column names as keys and encoding type as values.
                        Types: 'ordinal', 'one-hot', 'none'
        """
        for col, encoding in columns.items():
            if encoding == 'ordinal':
                self.df[col] = pd.factorize(self.df[col])[0]
            elif encoding == 'one-hot':
                one_hot = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, one_hot], axis=1)
                self.df = self.df.drop(col, axis=1)
        return self

    def scale_numerical(self, columns: List[str], method: str = 'standard'):
        """
        Scale numerical features.
        
        :param columns: List of columns to scale
        :param method: Scaling method ('standard', 'minmax', 'robust')
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaling method")

        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self

    def create_polynomial_features(self, columns: List[str], degree: int = 2):
        """
        Create polynomial features for specified columns.
        
        :param columns: List of columns to create polynomial features for
        :param degree: Degree of the polynomial features
        """
        for col in columns:
            for d in range(2, degree + 1):
                self.df[f"{col}^{d}"] = self.df[col] ** d
        return self

    def create_interaction_features(self, columns: List[str]):
        """
        Create interaction features between specified columns.
        
        :param columns: List of columns to create interactions for
        """
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                self.df[f"{col1}_{col2}_interaction"] = self.df[col1] * self.df[col2]
        return self

    def bin_numerical(self, column: str, bins: int, labels: Optional[List] = None):
        """
        Bin a numerical column into categories.
        
        :param column: Column to bin
        :param bins: Number of bins or list of bin edges
        :param labels: Labels for the bins
        """
        self.df[f"{column}_binned"] = pd.cut(self.df[column], bins=bins, labels=labels)
        return self

    def create_date_features(self, column: str):
        """
        Create date-related features from a date column.
        
        :param column: Date column to extract features from
        """
        self.df[column] = pd.to_datetime(self.df[column])
        self.df[f"{column}_year"] = self.df[column].dt.year
        self.df[f"{column}_month"] = self.df[column].dt.month
        self.df[f"{column}_day"] = self.df[column].dt.day
        self.df[f"{column}_dayofweek"] = self.df[column].dt.dayofweek
        self.df[f"{column}_quarter"] = self.df[column].dt.quarter
        return self

    def apply_log_transform(self, columns: List[str]):
        """
        Apply log transformation to specified columns.
        
        :param columns: List of columns to transform
        """
        for col in columns:
            self.df[f"{col}_log"] = np.log1p(self.df[col])
        return self

    def apply_box_cox_transform(self, columns: List[str]):
        """
        Apply Box-Cox transformation to specified columns.
        
        :param columns: List of columns to transform
        """
        from scipy.stats import boxcox
        for col in columns:
            self.df[f"{col}_boxcox"], _ = boxcox(self.df[col] + 1)  # Adding 1 to handle zero values
        return self

    def remove_low_variance_features(self, threshold: float = 0.1):
        """
        Remove features with low variance.
        
        :param threshold: Variance threshold
        """
        selector = VarianceThreshold(threshold)
        selected_features = selector.fit_transform(self.df)
        self.df = pd.DataFrame(selected_features, columns=self.df.columns[selector.get_support()])
        return self

    def apply_pca(self, n_components: Union[int, float, str] = 0.95):
        """
        Apply Principal Component Analysis (PCA).
        
        :param n_components: Number of components to keep
        """
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(self.df)
        self.df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
        return self

    def get_feature_importance(self, target: str, model=None):
        """
        Get feature importance using a tree-based model.
        
        :param target: Target variable
        :param model: Model to use for feature importance (default: RandomForestRegressor)
        """
        from sklearn.ensemble import RandomForestRegressor
        if model is None:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        X = self.df.drop(target, axis=1)
        y = self.df[target]
        model.fit(X, y)
        
        importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
        return importance.sort_values('importance', ascending=False)

    def get_engineered_data(self):
        """Return the engineered DataFrame."""
        return self.df

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    df = pd.read_csv("input/train_folds.csv")

    fe = FeatureEngineer(df)

    # Apply various feature engineering techniques
    fe.handle_missing_values({'age': 'mean', 'income': 'median'})
    fe.encode_categorical({'gender': 'one-hot', 'category': 'ordinal'})
    fe.scale_numerical(['age', 'income'], method='standard')
    fe.create_polynomial_features(['age', 'income'], degree=2)
    fe.create_interaction_features(['age', 'income'])
    fe.bin_numerical('age', bins=3, labels=['young', 'middle', 'old'])
    fe.create_date_features('date')
    fe.apply_log_transform(['income'])

    # Get the engineered data
    engineered_df = fe.get_engineered_data()
    engineered_df.to_csv("input/engineered_data.csv", index=False)
    print("Original DataFrame:")
    print(df.head())
    print("\nEngineered DataFrame:")
    print(engineered_df.head())
    print("\nNew features created:", set(engineered_df.columns) - set(df.columns))

    