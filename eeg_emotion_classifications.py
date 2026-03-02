"""
Experiment 2: Performance Comparison - Emotion Recognition Methods

This script compares different approaches to emotion recognition from EEG data.
Measure execution time and accuracy for each method.
"""

import time
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class EEGEmotionClassifier:
    """Base class for emotion classification from EEG"""
    
    def __init__(self, sampling_rate=256):
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        self.model = None
        
    def extract_features_basic(self, eeg_signal):
        """Basic feature extraction - statistical only"""
        features = []
        for channel in range(eeg_signal.shape[1]):
            signal_channel = eeg_signal[:, channel]
            features.append(np.mean(signal_channel))
            features.append(np.std(signal_channel))
            features.append(np.var(signal_channel))
        return np.array(features)
    
    def extract_features_advanced(self, eeg_signal):
        """Advanced feature extraction - frequency domain"""
        features = []
        for channel in range(eeg_signal.shape[1]):
            signal_channel = eeg_signal[:, channel]
            
            # Statistical features
            features.append(np.mean(signal_channel))
            features.append(np.std(signal_channel))
            features.append(np.var(signal_channel))
            
            # Frequency domain features
            try:
                freq, psd = signal.welch(signal_channel, fs=self.sampling_rate, nperseg=256)
                
                # Power in frequency bands
                alpha_power = np.sum(psd[(freq > 8) & (freq < 12)])
                beta_power = np.sum(psd[(freq > 12) & (freq < 30)])
                gamma_power = np.sum(psd[(freq > 30) & (freq < 50)])
                
                features.extend([alpha_power, beta_power, gamma_power])
                
                # Band ratios
                total_power = np.sum(psd)
                if total_power > 0:
                    features.append(alpha_power / total_power)
                    features.append(beta_power / total_power)
            except:
                # If frequency analysis fails, skip
                features.extend([0, 0, 0, 0, 0])
        
        return np.array(features)


class Method1_BasicFeatures:
    """Method 1: Simple statistical features + Logistic Regression"""
    
    def __init__(self, sampling_rate=256):
        self.classifier = EEGEmotionClassifier(sampling_rate)
        self.model = LogisticRegression(max_iter=1000)
        self.time_taken = 0
        
    def train(self, X_train, y_train):
        """Train the classifier"""
        start = time.time()
        
        # Extract basic features
        X_features = np.array([self.classifier.extract_features_basic(x) for x in X_train])
        
        # Scale and train
        X_scaled = self.classifier.scaler.fit_transform(X_features)
        self.model.fit(X_scaled, y_train)
        
        self.time_taken = time.time() - start
        
    def predict(self, X_test):
        """Make predictions"""
        X_features = np.array([self.classifier.extract_features_basic(x) for x in X_test])
        X_scaled = self.classifier.scaler.transform(X_features)
        return self.model.predict(X_scaled)
    
    def get_info(self):
        return {
            'method': 'Method 1: Basic Features + Logistic Regression',
            'features': 'Statistical only (mean, std, var)',
            'classifier': 'Logistic Regression',
            'training_time': self.time_taken
        }


class Method2_AdvancedFeatures:
    """Method 2: Advanced frequency features + Random Forest"""
    
    def __init__(self, sampling_rate=256):
        self.classifier = EEGEmotionClassifier(sampling_rate)
        self.model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        self.time_taken = 0
        
    def train(self, X_train, y_train):
        """Train the classifier"""
        start = time.time()
        
        # Extract advanced features
        X_features = np.array([self.classifier.extract_features_advanced(x) for x in X_train])
        
        # Scale and train
        X_scaled = self.classifier.scaler.fit_transform(X_features)
        self.model.fit(X_scaled, y_train)
        
        self.time_taken = time.time() - start
        
    def predict(self, X_test):
        """Make predictions"""
        X_features = np.array([self.classifier.extract_features_advanced(x) for x in X_test])
        X_scaled = self.classifier.scaler.transform(X_features)
        return self.model.predict(X_scaled)
    
    def get_info(self):
        return {
            'method': 'Method 2: Advanced Features + Random Forest',
            'features': 'Statistical + Frequency domain (alpha/beta/gamma power)',
            'classifier': 'Random Forest (50 trees)',
            'training_time': self.time_taken
        }


def run_comparison(X_train, X_test, y_train, y_test):
    """Compare two methods on the same data"""
    
    print("="*70)
    print("EXPERIMENT 2: METHOD COMPARISON")
    print("="*70)
    
    results = []
    
    # Method 1: Basic features + Logistic Regression
    print("\nTesting Method 1: Basic Features + Logistic Regression...")
    method1 = Method1_BasicFeatures()
    method1.train(X_train, y_train)
    y_pred_1 = method1.predict(X_test)
    acc_1 = accuracy_score(y_test, y_pred_1)
    info_1 = method1.get_info()
    info_1['test_accuracy'] = acc_1
    results.append(info_1)
    print(f"  Training time: {info_1['training_time']:.4f} seconds")
    print(f"  Test accuracy: {acc_1:.4f}")
    
    # Method 2: Advanced features + Random Forest
    print("\nTesting Method 2: Advanced Features + Random Forest...")
    method2 = Method2_AdvancedFeatures()
    method2.train(X_train, y_train)
    y_pred_2 = method2.predict(X_test)
    acc_2 = accuracy_score(y_test, y_pred_2)
    info_2 = method2.get_info()
    info_2['test_accuracy'] = acc_2
    results.append(info_2)
    print(f"  Training time: {info_2['training_time']:.4f} seconds")
    print(f"  Test accuracy: {acc_2:.4f}")
    
    # Create comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    
    comparison_df = pd.DataFrame({
        'Method': [r['method'] for r in results],
        'Features': [r['features'] for r in results],
        'Classifier': [r['classifier'] for r in results],
        'Training Time (s)': [f"{r['training_time']:.4f}" for r in results],
        'Test Accuracy': [f"{r['test_accuracy']:.4f}" for r in results]
    })
    
    print(comparison_df.to_string(index=False))
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    time_diff = abs(info_1['training_time'] - info_2['training_time'])
    acc_diff = abs(acc_1 - acc_2)
    
    print(f"\nTime Difference: {time_diff:.4f} seconds")
    if info_1['training_time'] < info_2['training_time']:
        print(f"Method 1 is {time_diff/info_2['training_time']*100:.1f}% faster")
    else:
        print(f"Method 2 is {time_diff/info_1['training_time']*100:.1f}% faster")
    
    print(f"\nAccuracy Difference: {acc_diff:.4f}")
    if acc_1 > acc_2:
        print(f"Method 1 is more accurate by {acc_diff*100:.1f} percentage points")
    else:
        print(f"Method 2 is more accurate by {acc_diff*100:.1f} percentage points")
    
    print("\nTrade-offs:")
    print("- Method 1: Faster, simpler, uses fewer features")
    print("- Method 2: More comprehensive features, may be slower but potentially more accurate")
    
    return comparison_df, results


# TODO: Run this with your data
# if __name__ == "__main__":
#     # Load your data
#     X_train = ...  # Your training data
#     X_test = ...   # Your testing data
#     y_train = ...  # Training labels
#     y_test = ...   # Testing labels
#     
#     # Run comparison
#     comparison_table, results = run_comparison(X_train, X_test, y_train, y_test)
#     
#     # Save results
#     comparison_table.to_csv('method_comparison.csv', index=False)
#     print("\nResults saved to method_comparison.csv")
#     
#     # Document findings in Word document
#     print("\nCopy this table into your Word document for Experiment 2")
