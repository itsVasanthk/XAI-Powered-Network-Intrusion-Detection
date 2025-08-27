"""
Simple test script to verify NSL-KDD data loading
Run this first to make sure your dataset is properly downloaded and accessible
"""

from data_preprocessing import load_nsl_kdd, preprocess_data
import os

def test_file_paths():
    """Check if the dataset files exist"""
    print("=== Testing File Paths ===")
    
    # Try to import config first
    try:
        from nsl_kdd_config import TRAIN_FILE, TEST_FILE
        files_to_check = [TRAIN_FILE, TEST_FILE]
        print("✅ Using paths from nsl_kdd_config.py")
    except ImportError:
        print("⚠️  Config file not found, using default paths")
        files_to_check = [
            "data/NSL-KDD/KDDTrain+.txt",
            "data/NSL-KDD/KDDTest+.txt", 
            "data/NSL-KDD/KDDTrain+_20Percent.txt"
        ]
    
    all_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            all_exist = False
    
    return all_exist

def test_data_loading():
    """Test loading a small sample of data"""
    print("\n=== Testing Data Loading ===")
    
    try:
        # Try to get paths from config
        try:
            from nsl_kdd_config import TRAIN_FILE, TEST_FILE
            train_path = TRAIN_FILE
            test_path = TEST_FILE
        except ImportError:
            # Use default paths
            train_path = "data/NSL-KDD/KDDTrain+_20Percent.txt"
            test_path = "data/NSL-KDD/KDDTest+.txt"
            
            # If 20% doesn't exist, try full file
            if not os.path.exists(train_path):
                train_path = "data/NSL-KDD/KDDTrain+.txt"
        
        print("Loading data...")
        train_df, test_df = load_nsl_kdd(train_path, test_path)
        
        print(f"✓ Training data shape: {train_df.shape}")
        print(f"✓ Testing data shape: {test_df.shape}")
        
        # Show sample data
        print(f"\nSample of training data:")
        print(train_df.head(3))
        
        # Check attack types
        print(f"\nUnique attack types in training data:")
        print(train_df['attack_type'].value_counts().head(10))
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

def test_preprocessing():
    """Test data preprocessing"""
    print("\n=== Testing Data Preprocessing ===")
    
    try:
        # Try to get paths from config
        try:
            from nsl_kdd_config import TRAIN_FILE, TEST_FILE
            train_path = TRAIN_FILE
            test_path = TEST_FILE
        except ImportError:
            train_path = "data/NSL-KDD/KDDTrain+_20Percent.txt"
            test_path = "data/NSL-KDD/KDDTest+.txt"
            
            if not os.path.exists(train_path):
                train_path = "data/NSL-KDD/KDDTrain+.txt"
        
        # Load data
        train_df, test_df = load_nsl_kdd(train_path, test_path)
        
        # Preprocess
        print("Preprocessing data...")
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(train_df, test_df)
        
        print(f"✓ Processed training features shape: {X_train.shape}")
        print(f"✓ Processed testing features shape: {X_test.shape}")
        print(f"✓ Number of features: {len(feature_names)}")
        print(f"✓ Training labels shape: {y_train.shape}")
        print(f"✓ Testing labels shape: {y_test.shape}")
        
        # Check class distribution
        print(f"\nClass distribution in training data:")
        print(f"Normal (0): {sum(y_train == 0)}")
        print(f"Attack (1): {sum(y_train == 1)}")
        
        print(f"\nClass distribution in testing data:")
        print(f"Normal (0): {sum(y_test == 0)}")
        print(f"Attack (1): {sum(y_test == 1)}")
        
        print(f"\nFirst 10 feature names:")
        for i, name in enumerate(feature_names[:10]):
            print(f"{i+1}. {name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in preprocessing: {e}")
        return False

def main():
    """Run all tests"""
    print("NSL-KDD Dataset Loading Test")
    print("=" * 40)
    
    # Test 1: Check file existence
    files_exist = test_file_paths()
    
    if not files_exist:
        print("\n❌ FAILED: Some dataset files are missing!")
        print("\nPlease ensure you have downloaded:")
        print("1. KDDTrain+.txt")
        print("2. KDDTest+.txt") 
        print("3. KDDTrain+_20Percent.txt (if available)")
        print("\nAnd run: python setup_kaggle_data.py")
        print("This will help organize your files correctly.")
        return
    
    # Test 2: Load data
    loading_success = test_data_loading()
    
    if not loading_success:
        print("\n❌ FAILED: Could not load data!")
        print("Try running: python setup_kaggle_data.py")
        return
    
    # Test 3: Preprocess data
    preprocessing_success = test_preprocessing()
    
    if not preprocessing_success:
        print("\n❌ FAILED: Could not preprocess data!")
        return
    
    print("\n" + "=" * 40)
    print("✅ ALL TESTS PASSED!")
    print("Your dataset is ready for XAI analysis.")
    print("You can now run: python ids_xai.py")

if __name__ == "__main__":
    main()