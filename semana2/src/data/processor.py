import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SocialAdDataProcessor:
    """
    Handles loading, cleaning, and preprocessing of the Social Network Ads dataset.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Loads data from CSV."""
        print(f"Loading data from {self.filepath}...")
        self.df = pd.read_csv(self.filepath)
        print("Data loaded successfully.")
        return self.df
    
    def preprocess(self):
        """
        Cleans data, encodes categorical variables, and defines features (X) and target (y).
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # 1. Drop unnecessary columns
        if 'User ID' in self.df.columns:
            self.df = self.df.drop('User ID', axis=1)
            
        # 2. Encode Gender if present
        if 'Gender' in self.df.columns:
            self.df['Gender'] = self.df['Gender'].map({'Female': 0, 'Male': 1})
            self.X = self.df[['Gender', 'Age', 'EstimatedSalary']]
        else:
            self.X = self.df[['Age', 'EstimatedSalary']]
            
        self.y = self.df['Purchased']
        
        # 3. Scale features
        print("Scaling features...")
        self.X_scaled = self.scaler.fit_transform(self.X)
        
    def get_data_splits(self, test_size=0.20, random_state=42):
        """
        Splits data into training and testing sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state
        )
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
