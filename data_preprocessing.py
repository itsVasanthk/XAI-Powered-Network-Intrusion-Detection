import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_nsl_kdd(train_path, test_path):
    """
    Load and preprocess NSL-KDD dataset (works with various formats)
    
    Args:
        train_path: Path to training file (KDDTrain+.txt)
        test_path: Path to testing file (KDDTest+.txt)
    
    Returns:
        train_df, test_df: Loaded dataframes
    """
    # NSL-KDD column names (41 features + attack_type + difficulty_level)
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count',
        'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
    ]
    
    print("Loading training data...")
    try:
        # Try loading with comma separator first
        train_df = pd.read_csv(train_path, names=columns, header=None)
        
        # Check if data loaded correctly by examining shape and first row
        if train_df.shape[1] != len(columns):
            print(f"Warning: Expected {len(columns)} columns, got {train_df.shape[1]}")
            # Try loading without difficulty_level column
            columns_without_difficulty = columns[:-1]  # Remove last column
            train_df = pd.read_csv(train_path, names=columns_without_difficulty, header=None)
            
    except Exception as e:
        print(f"Error loading training data: {e}")
        print("Trying alternative loading method...")
        # Try with different separator or encoding
        train_df = pd.read_csv(train_path, names=columns, header=None, sep=',', encoding='utf-8')
    
    print("Loading testing data...")
    try:
        test_df = pd.read_csv(test_path, names=columns, header=None)
        
        # Check if data loaded correctly
        if test_df.shape[1] != len(columns):
            print(f"Warning: Expected {len(columns)} columns, got {test_df.shape[1]}")
            # Try loading without difficulty_level column
            columns_without_difficulty = columns[:-1]
            test_df = pd.read_csv(test_path, names=columns_without_difficulty, header=None)
            
    except Exception as e:
        print(f"Error loading testing data: {e}")
        print("Trying alternative loading method...")
        test_df = pd.read_csv(test_path, names=columns, header=None, sep=',', encoding='utf-8')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")
    
    # Show sample of loaded data to verify
    print("\nSample of training data (first 2 rows):")
    print(train_df.head(2))
    
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """
    Preprocess the network data for ML models
    
    Args:
        train_df, test_df: Raw dataframes from NSL-KDD
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print("Starting data preprocessing...")
    
    # Remove difficulty level column (not needed for classification)
    if 'difficulty_level' in train_df.columns:
        train_df = train_df.drop('difficulty_level', axis=1)
        test_df = test_df.drop('difficulty_level', axis=1)
    
    # Create binary classification labels (Normal vs Attack)
    print("Creating binary labels...")
    train_df['binary_class'] = train_df['attack_type'].apply(
        lambda x: 0 if x == 'normal' else 1
    )
    test_df['binary_class'] = test_df['attack_type'].apply(
        lambda x: 0 if x == 'normal' else 1
    )
    
    print("Attack distribution in training data:")
    print(train_df['binary_class'].value_counts())
    print("\nAttack distribution in testing data:")
    print(test_df['binary_class'].value_counts())
    
    # Handle categorical variables
    print("Encoding categorical variables...")
    categorical_columns = ['protocol_type', 'service', 'flag']
    
    # Combine train and test for consistent encoding
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col])
        label_encoders[col] = le
    
    # Split back to train and test
    train_processed = combined_df[:len(train_df)].copy()
    test_processed = combined_df[len(train_df):].copy()
    
    # Separate features and target
    feature_columns = [col for col in train_processed.columns 
                      if col not in ['attack_type', 'binary_class']]
    
    X_train = train_processed[feature_columns]
    y_train = train_processed['binary_class']
    X_test = test_processed[feature_columns]
    y_test = test_processed['binary_class']
    
    print(f"Features shape: {X_train.shape}")
    print(f"Number of features: {len(feature_columns)}")
    
    # Handle any missing values
    print("Checking for missing values...")
    if X_train.isnull().sum().sum() > 0:
        print("Found missing values, imputing...")
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_columns)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)
    
    print("Preprocessing completed!")
    print(f"Final training set shape: {X_train_scaled.shape}")
    print(f"Final testing set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns

def get_attack_types_info(train_df, test_df):
    """
    Get information about different attack types in the dataset
    """
    print("\n=== Attack Types Analysis ===")
    print("Training data attack types:")
    print(train_df['attack_type'].value_counts())
    print("\nTesting data attack types:")
    print(test_df['attack_type'].value_counts())
    
    # Attack categories
    attack_categories = {
        'normal': 'Normal',
        'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
        'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R', 'ps': 'U2R',
        'sqlattack': 'U2R', 'xterm': 'U2R', 'httptunnel': 'U2R',
        'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L',
        'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
        'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'worm': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L',
        'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 'mscan': 'Probe',
        'saint': 'Probe'
    }
    
    return attack_categories

# Test function to verify data loading
def test_data_loading():
    """Test function to verify data loading works correctly"""
    try:
        # Update these paths to match your file locations
        train_path = "data/NSL-KDD/KDDTrain+_20Percent.txt"  # Using 20% for testing
        test_path = "data/NSL-KDD/KDDTest+.txt"
        
        # Load data
        train_df, test_df = load_nsl_kdd(train_path, test_path)
        
        # Show basic info
        print(f"Training data loaded successfully: {train_df.shape}")
        print(f"Testing data loaded successfully: {test_df.shape}")
        print(f"Sample of training data:")
        print(train_df.head())
        
        # Get attack types info
        get_attack_types_info(train_df, test_df)
        
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

if __name__ == "__main__":
    # Run test
    test_data_loading()