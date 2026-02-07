import pandas as pd
from sklearn.preprocessing import StandardScaler


class UserBehaviorDataProcessor:
    """
    Handles loading, cleaning, and preprocessing of the User Behavior dataset.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.X = None
        self.X_scaled = None
        self.selected_variables = None
        self.scaler = StandardScaler()

    def load_data(self):
        """Loads data from CSV and prints exploratory info."""
        print(f"Loading data from {self.filepath}...")
        self.df = pd.read_csv(self.filepath)
        print("Data loaded successfully.")
        print(self.df.head())
        print(self.df.info())
        print(self.df.describe())
        return self.df

    def preprocess(self):
        """
        Validates data quality, creates derived columns,
        selects features, and scales them.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Validate nulls
        print("\nValidar nulos:")
        print(self.df.isnull().sum())

        # Validate duplicates
        print("\nValidar duplicados:")
        duplicados_id = self.df.duplicated(subset=['User ID']).sum()
        print(f"Duplicados por User ID: {duplicados_id}")

        # Create derived column
        self.df['App Usage Time (hours/day)'] = self.df['App Usage Time (min/day)'] / 60

        # Select features for clustering
        self.selected_variables = [
            'Age',
            'Screen On Time (hours/day)',
            'Data Usage (MB/day)',
            'Number of Apps Installed'
        ]
        self.X = self.df[self.selected_variables]

        # Scale features
        print("\nScaling features...")
        self.X_scaled = self.scaler.fit_transform(self.X)
        print(f"Feature matrix shape: {self.X_scaled.shape}")

    def get_scaled_data(self):
        """Returns the scaled feature matrix."""
        if self.X_scaled is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        return self.X_scaled
