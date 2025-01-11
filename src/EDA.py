import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns

NULL_PERCENTAGE = 0.25



'''plotting features by null percentage'''
df = pd.read_csv('data/raw/Dev_data_to_be_shared.csv')
null_percentage = df.isnull().mean() * 100
# Define bins for null percentage ranges
bins = [0, 10, 20,30,40,50,60,70,80,90,100]
labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

# Bin the features based on null percentage
categories = pd.cut(null_percentage, bins=bins, labels=labels, include_lowest=True)

# Count the number of features in each bin
feature_counts = categories.value_counts().sort_index()

# Plot the graph
plt.figure(figsize=(8, 5))
feature_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Features by Null Percentage')
plt.xlabel('Null Percentage Range')
plt.ylabel('Number of Features')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('assets/features_by_null_percentage.png')


'''plotting Number of features with null values greater than threshold'''
df = pd.read_csv('data/raw/Dev_data_to_be_shared.csv')
# Calculate the percentage of null values for each feature
null_percentage = df.isnull().mean() * 100

# Define thresholds for null percentages
thresholds = np.arange(0, 105, 5)

# Count features exceeding each threshold
feature_counts = [sum(null_percentage > t) for t in thresholds]

# Plot the graph
plt.figure(figsize=(8, 5))
plt.plot(thresholds, feature_counts, marker='o', linestyle='-', color='blue', label='Features Exceeding Threshold')
plt.title('Number of Features with Null Values Greater than Threshold')
plt.xlabel('Threshold Percentage')
plt.ylabel('Number of Features')
plt.xticks(thresholds)
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('assets/num_features_threshold.png')



'''plotting the variance in features'''
df = pd.read_csv('data/raw/Dev_data_to_be_shared.csv')
df = df.loc[:, df.isnull().mean()  <= NULL_PERCENTAGE]
# Set Seaborn style
sns.set(style="whitegrid", palette="muted")

thresholds = np.arange(0, 0.55, 0.01)

# Store the number of features retained for each threshold
num_features = []

for threshold in thresholds:
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    num_features.append(df.shape[1] - selector.get_support().sum())
    
# Plot the results
plt.figure(figsize=(10, 6))

plt.text(0, num_features[0], f'{num_features[0]}', color='darkred', fontsize=10, ha='right', va='bottom')
plt.text(0.01, num_features[1], f'{num_features[1]}', color='darkred', fontsize=10, ha='right', va='bottom')
plt.text(0.02, num_features[2], f'{num_features[2]}', color='darkred', fontsize=10, ha='right', va='bottom')
plt.text(0.03, num_features[3], f'{num_features[3]}', color='darkred', fontsize=10, ha='right', va='bottom')
plt.text(0.04, num_features[4], f'{num_features[4]}', color='darkred', fontsize=10, ha='right', va='bottom')
plt.text(0.05, num_features[5], f'{num_features[5]}', color='darkred', fontsize=10, ha='right', va='bottom')

        
plt.plot(thresholds, num_features, marker='o', linestyle='-', color='blue')
plt.title('Number of Features vs Variance Threshold')
plt.xlabel('Variance Threshold')
plt.ylabel(f'Number of Features Removed')
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('assets/num_features_variance.png')