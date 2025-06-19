import matplotlib.pyplot as plt

class DataVisualization:

    def visualize_class_distribution(self, y, labels=["ham", "spam"], type="bar", save=True, save_folder="./figs"):
        """
        Visualize the class distribution of the target variable.

        Parameters:
        type (str): Type of plot to use ('bar' or 'pie').
        save (bool): Whether to save the plot.
        save_folder (str): Folder to save the plot if save is True.

        Returns:
        None
        """
        if type == "bar":
            plt.figure(figsize=(10, 6))
            y.value_counts().plot(kind='bar')
            plt.title(f'Class Distribution in {self.name}')
            plt.xlabel('Classes')
            plt.ylabel('Frequency')
            if save:
                plt.savefig(f"{save_folder}/{self.name}_class_distribution.png")
            plt.show()
        elif type == "pie":
            plt.figure(figsize=(8, 8))
            plt.pie(y.values, labels=labels, autopct='%1.1f%%', startangle=140)
            plt.title(f'Class Distribution')
            plt.ylabel('')
            if save:
                plt.savefig(f"{save_folder}/{self.name}_class_distribution_pie.png")
            plt.show()
        else:
            raise ValueError("Unsupported plot type. Use 'bar' or 'pie'.")
        
    def visualize_length_distribution(self, X, X_preprocessed, y, save=True, save_folder="./figs"):
        """
        Visualize the distribution of text lengths in the dataset.

        Parameters:
        X (pd.Series): Original text data.
        X_preprocessed (pd.Series): Preprocessed text data.
        y (pd.Series): Target variable.
        save (bool): Whether to save the plot.
        save_folder (str): Folder to save the plot if save is True.

        Returns:
        None
        """
        plt.figure(figsize=(12, 6))
        plt.hist(X.str.len(), bins=50, alpha=0.5, label='Original Text Lengths')
        plt.hist(X_preprocessed.str.len(), bins=50, alpha=0.5, label='Preprocessed Text Lengths')
        plt.title(f'Text Length Distribution in {self.name}')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.legend()
        if save:
            plt.savefig(f"{save_folder}/{self.name}_text_length_distribution.png")
        plt.show()