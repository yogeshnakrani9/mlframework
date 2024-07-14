import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class Visualizer:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_columns = self.df.select_dtypes(include=['object', 'bool', 'category']).columns


    def show_basic_info(self):
        print(self.df.info())
        print("\nSummary Statistics:")
        print(self.df.describe())

    def plot_histogram(self, column: str, bins: int = 30):
        plt.figure(figsize=(10, 6))
        self.df[column].hist(bins=bins)
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()
    
    def plot_boxplot(self, column: str):
        plt.figure(figsize=(10, 6))
        self.df.boxplot(column=column)
        plt.title(f"Boxplot of {column}")
        plt.ylabel(column)
        plt.show()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm")
        plt.title('Correlation Heatmap')
        plt.show()

    def plot_scatter_plot(self, x_column: str, y_column: str):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df[x_column], self.df[y_column])
        plt.title(f"Scatter Plot of {x_column} vs. {y_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()


    def plot_box_cox(self, column: str):
        data = self.df[column]
        transformed_data, lambda_params = stats.boxcox(data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.hist(data, bins = 30)
        ax1.set_title(f'Original {column}')

        ax2.hist(transformed_data, bins = 30)
        ax2.set_title(f'Box-Cox Transformed {column} (λ = {lambda_params:.2f})')

        plt.tight_layout()
        plt.show()
