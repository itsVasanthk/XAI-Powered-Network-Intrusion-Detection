import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns

# Import our data preprocessing module
from data_preprocessing import load_nsl_kdd, preprocess_data, get_attack_types_info

def main():
    """Main function to run the XAI-powered IDS"""
    
    print("=== XAI-Powered Network Intrusion Detection System ===\n")
    
    # Step 1: Load the NSL-KDD dataset
    print("Step 1: Loading NSL-KDD Dataset...")
    
    # Try to import config first, then fallback to default paths
    try:
        from nsl_kdd_config import TRAIN_FILE, TEST_FILE
        train_path = TRAIN_FILE
        test_path = TEST_FILE
        print(f"✅ Using paths from config file")
    except ImportError:
        print("⚠️  Config file not found, using default paths")
        train_path = "data/NSL-KDD/KDDTrain+_20Percent.txt"  # Using 20% for faster processing
        test_path = "data/NSL-KDD/KDDTest+.txt"
        
        # Check if 20% file exists, if not use full file
        import os
        if not os.path.exists(train_path):
            train_path = "data/NSL-KDD/KDDTrain+.txt"
    
    print(f"Training file: {train_path}")
    print(f"Testing file: {test_path}")
    
    try:
        # Load raw data
        train_df, test_df = load_nsl_kdd(train_path, test_path)
        
        # Get attack types information
        attack_categories = get_attack_types_info(train_df, test_df)
        
        # Step 2: Preprocess the data
        print("\nStep 2: Preprocessing Data...")
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(train_df, test_df)
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find dataset files.")
        print(f"Please run: python setup_kaggle_data.py first")
        print(f"Expected files:")
        print(f"- {train_path}")
        print(f"- {test_path}")
        return
    except Exception as e:
        print(f"❌ Error loading/preprocessing data: {e}")
        return
    
    # Step 3: Train the Random Forest model
    print("\nStep 3: Training Random Forest Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Step 4: Evaluate the model
    print("\nStep 4: Model Evaluation...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], 
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix - Network Intrusion Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Step 5: SHAP Analysis
    print("\nStep 5: SHAP Explainability Analysis...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a subset of test data (for faster computation)
    shap_sample_size = min(1000, len(X_test))
    X_test_sample = X_test.iloc[:shap_sample_size]
    y_test_sample = y_test.iloc[:shap_sample_size]
    
    print(f"Calculating SHAP values for {shap_sample_size} samples...")
    shap_values = explainer.shap_values(X_test_sample)
    
    # SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values[1], X_test_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - Feature Importance for Attack Detection')
    plt.tight_layout()
    plt.show()
    
    # Feature Importance from SHAP
    feature_importance = np.abs(shap_values[1]).mean(0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 10 Most Important Features (SHAP):")
    print("=" * 50)
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<25} {row['importance']:.6f}")
    
    # Plot top features
    plt.figure(figsize=(12, 6))
    top_features = importance_df.head(10)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Mean |SHAP Value| (Feature Importance)')
    plt.title('Top 10 Most Important Features for Attack Detection')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.0001, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Step 6: LIME Analysis
    print("\nStep 6: LIME Analysis for Individual Predictions...")
    
    # Convert to numpy for LIME
    X_train_np = X_train.values
    X_test_np = X_test.values
    
    # Create LIME explainer
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=feature_names,
        class_names=['Normal', 'Attack'],
        mode='classification',
        discretize_continuous=True
    )
    
    # Explain a few different instances
    instances_to_explain = [0, 100, 200, 500, 800]  # Different test instances
    
    for i in instances_to_explain:
        if i < len(X_test):
            print(f"\n💡 Explaining prediction for test instance {i}:")
            true_label = 'Attack' if y_test.iloc[i] == 1 else 'Normal'
            pred_label = 'Attack' if y_pred[i] == 1 else 'Normal'
            confidence = model.predict_proba(X_test_np[i:i+1])[0].max()
            
            print(f"   True label: {true_label}")
            print(f"   Predicted label: {pred_label}")
            print(f"   Confidence: {confidence:.3f}")
            
            # Get LIME explanation
            try:
                exp = explainer_lime.explain_instance(
                    X_test_np[i], 
                    model.predict_proba,
                    num_features=10
                )
                
                # Save explanation as HTML
                filename = f'lime_explanation_instance_{i}_{true_label.lower()}.html'
                exp.save_to_file(filename)
                print(f"   📄 Saved explanation: {filename}")
                
            except Exception as e:
                print(f"   ⚠️  Error generating LIME explanation: {e}")
    
    # Step 7: Attack Pattern Analysis
    print("\nStep 7: Attack Pattern Analysis...")
    
    # Find attack instances in test set
    attack_indices = np.where(y_test_sample == 1)[0]
    normal_indices = np.where(y_test_sample == 0)[0]
    
    if len(attack_indices) > 0:
        print(f"🎯 Found {len(attack_indices)} attack instances in sample")
        print(f"📊 Found {len(normal_indices)} normal instances in sample")
        
        # Average SHAP values for attacks vs normal
        attack_shap_mean = np.abs(shap_values[1][attack_indices]).mean(0)
        normal_shap_mean = np.abs(shap_values[1][normal_indices]).mean(0)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'feature': feature_names,
            'attack_importance': attack_shap_mean,
            'normal_importance': normal_shap_mean
        })
        comparison_df['difference'] = comparison_df['attack_importance'] - comparison_df['normal_importance']
        comparison_df = comparison_df.sort_values('difference', ascending=False)
        
        print("\n🚨 Features most distinctive for attacks:")
        print("-" * 60)
        for i, (_, row) in enumerate(comparison_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<20} | Diff: {row['difference']:8.4f} | Attack: {row['attack_importance']:.4f} | Normal: {row['normal_importance']:.4f}")
        
        # Plot comparison
        plt.figure(figsize=(14, 8))
        top_diff_features = comparison_df.head(10)
        
        x = np.arange(len(top_diff_features))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, top_diff_features['attack_importance'], width, 
                       label='Attack Instances', alpha=0.8, color='red')
        bars2 = plt.bar(x + width/2, top_diff_features['normal_importance'], width, 
                       label='Normal Instances', alpha=0.8, color='blue')
        
        plt.xlabel('Features')
        plt.ylabel('Mean |SHAP Value|')
        plt.title('Feature Importance: Attack vs Normal Traffic')
        plt.xticks(x, top_diff_features['feature'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Step 8: Model Performance Summary
    print("\n" + "=" * 70)
    print("🎉 XAI Analysis Complete!")
    print("=" * 70)
    print(f"📈 Final Model Accuracy: {accuracy:.4f}")
    print(f"🔢 Total features analyzed: {len(feature_names)}")
    print(f"📁 LIME explanations saved as HTML files")
    print(f"📊 SHAP visualizations displayed above")
    
    # Key insights
    print(f"\n📋 Key Insights:")
    print(f"• Most Important Feature: {importance_df.iloc[0]['feature']}")
    print(f"• Top 3 Attack Indicators: {', '.join(importance_df.head(3)['feature'].tolist())}")
    print(f"• Attack Detection Rate: {sum((y_test == 1) & (y_pred == 1)) / sum(y_test == 1):.3f}")
    print(f"• False Positive Rate: {sum((y_test == 0) & (y_pred == 1)) / sum(y_test == 0):.3f}")
    
    print(f"\n💡 Next Steps:")
    print(f"1. Review the SHAP plots to understand feature importance")
    print(f"2. Check the LIME HTML files for individual prediction explanations")
    print(f"3. Analyze the attack vs normal pattern differences")
    print(f"4. Consider feature engineering based on the insights")

if __name__ == "__main__":
    main()