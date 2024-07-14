import pandas as pd
from typing import List, Union, Dict

class CategoricalDataFrame:
    def __init__(self, df: pd.DataFrame, categorical_columns: Dict[str, str]):
        """
        Initialize with a DataFrame and a dictionary specifying categorical columns and their encoding.
        
        :param df: The input DataFrame
        :param categorical_columns: A dictionary where keys are column names and values are encoding types
                                    ('ordinal', 'one-hot', or 'none' for no encoding)
        """
        self.df = df.copy()
        self.categorical_columns = categorical_columns
        for col, encoding in categorical_columns.items():
            if encoding != 'none':
                self.df[col] = pd.Categorical(self.df[col])
    
    def get_categories(self, column: str) -> List[Union[str, int]]:
        """Return the unique categories in the specified column."""
        return list(self.df[column].cat.categories)
    
    def get_codes(self, column: str) -> pd.Series:
        """Return the numeric codes for each category in the specified column."""
        return self.df[column].cat.codes
    
    def encode(self) -> pd.DataFrame:
        """Encode categories based on the specified encoding for each column."""
        encoded_df = self.df.copy()
        
        for col, encoding in self.categorical_columns.items():
            if encoding == 'ordinal':
                encoded_df[col] = pd.factorize(self.df[col])[0]
            elif encoding == 'one-hot':
                one_hot = pd.get_dummies(self.df[col], prefix=col)
                encoded_df = pd.concat([encoded_df, one_hot], axis=1)
                encoded_df = encoded_df.drop(col, axis=1)
            # If encoding is 'none', we leave the column as is
        
        return encoded_df
    
    def frequency(self, column: str) -> pd.Series:
        """Return the frequency of each category in the specified column."""
        return self.df[column].value_counts()
    
    def add_category(self, column: str, category: Union[str, int]):
        """Add a new category to the existing categories in the specified column."""
        if self.categorical_columns[column] != 'none':
            self.df[column] = self.df[column].cat.add_categories([category])
    
    def remove_category(self, column: str, category: Union[str, int]):
        """Remove a category from the existing categories in the specified column."""
        if self.categorical_columns[column] != 'none':
            self.df[column] = self.df[column].cat.remove_categories([category])
    
    def rename_category(self, column: str, old_name: Union[str, int], new_name: Union[str, int]):
        """Rename a category in the specified column."""
        if self.categorical_columns[column] != 'none':
            self.df[column] = self.df[column].cat.rename_categories({old_name: new_name})

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    data = {
        'color': ['red', 'blue', 'green', 'red', 'yellow', 'blue'],
        'size': ['small', 'medium', 'large', 'medium', 'small', 'large'],
        'brand': ['A', 'B', 'C', 'A', 'B', 'C']
    }
    df = pd.DataFrame(data)
    
    # Specify encoding for each column
    categorical_columns = {
        'color': 'one-hot',
        'size': 'ordinal',
        'brand': 'none'
    }
    
    cat_df = CategoricalDataFrame(df, categorical_columns)
    
    print("Original DataFrame:\n", cat_df.df)
    print("\nEncoded DataFrame:\n", cat_df.encode())
    
    print("\nCategories in 'color':", cat_df.get_categories('color'))
    print("Codes for 'size':", cat_df.get_codes('size'))
    print("\nFrequency of 'brand':\n", cat_df.frequency('brand'))
    
    cat_df.add_category('color', 'purple')
    print("\nAfter adding 'purple' to 'color':", cat_df.get_categories('color'))
    
    cat_df.remove_category('size', 'small')
    print("\nAfter removing 'small' from 'size':", cat_df.get_categories('size'))
    
    cat_df.rename_category('color', 'green', 'forest')
    print("\nAfter renaming 'green' to 'forest' in 'color':", cat_df.get_categories('color'))