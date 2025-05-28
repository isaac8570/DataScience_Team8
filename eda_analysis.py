# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('preprocessed_dataset.csv')

# %% Histogram of Daily Screen Time Hours
plt.figure(figsize=(8, 6))
sns.histplot(df['daily_screen_time_hours'], bins=30, kde=True)
plt.title('Distribution of Daily Screen Time Hours')
plt.xlabel('Daily Screen Time (hours)')
plt.ylabel('Frequency')
plt.show()

# %% Outlier Detection - Boxplot for Numeric Features
numeric_cols = df.select_dtypes(include='number').columns

for i in range(0, len(numeric_cols), 5):
    cols_to_plot = numeric_cols[i:i+5]
    plt.figure(figsize=(18, 4))
    for j, col in enumerate(cols_to_plot):
        plt.subplot(1, 5, j+1)
        sns.boxplot(x=df[col])
        plt.title(col)
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# %% Feature Engineering - Derived Features
df['screen_to_sleep_ratio'] = df['daily_screen_time_hours'] / (df['sleep_duration_hours'] + 1e-5)
df['stress_per_hour_screen'] = df['stress_level'] / (df['daily_screen_time_hours'] + 1e-5)
df['active_age_ratio'] = df['physical_activity_hours_per_week'] / (df['age'] + 1e-5)

# %% Train/Test Split
from sklearn.model_selection import train_test_split

X = df.drop(['mental_health_score'], axis=1)
y = df['mental_health_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# %% Boxplot of Sleep Duration by Stress Level
plt.figure(figsize=(12, 6))
df['stress_level'] = df['stress_level'].astype(int)
ordered_levels = sorted(df['stress_level'].unique())
sns.boxplot(x='stress_level', y='sleep_duration_hours', data=df, order=ordered_levels)
plt.title('Sleep Duration by Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Sleep Duration (hours)')
plt.xticks(rotation=45)
plt.show()

# %% Full Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# %% Simplified Correlation Heatmap - Top Correlated with Mental Health Score
top_corr_features = df.corr()['mental_health_score'].abs().sort_values(ascending=False).head(10).index
plt.figure(figsize=(8, 6))
sns.heatmap(df[top_corr_features].corr(), cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Top Correlated Features with Mental Health Score')
plt.show()

# %% Scatterplot: Social Media Hours vs Mental Health Score
plt.figure(figsize=(8, 6))
sns.regplot(x='social_media_hours', y='mental_health_score', data=df, scatter_kws={'s': 10})
plt.title('Social Media Hours vs Mental Health Score')
plt.xlabel('Social Media Hours')
plt.ylabel('Mental Health Score')
plt.show()

# %% Boxplot: Healthy Eating vs Mental Health Score
plt.figure(figsize=(8, 6))
sns.boxplot(x='eats_healthy', y='mental_health_score', data=df)
plt.title('Mental Health Score by Healthy Eating')
plt.xlabel('Eats Healthy (0 = No, 1 = Yes)')
plt.ylabel('Mental Health Score')
plt.show()

# %% Boxplot: Wellness App Usage vs Stress Level
plt.figure(figsize=(8, 6))
sns.boxplot(x='uses_wellness_apps', y='stress_level', data=df)
plt.title('Stress Level by Wellness App Usage')
plt.xlabel('Uses Wellness Apps (0 = No, 1 = Yes)')
plt.ylabel('Stress Level')
plt.show()

# %% Scatterplot: Caffeine Intake vs Sleep Duration
plt.figure(figsize=(8, 6))
sns.regplot(x='caffeine_intake_mg_per_day', y='sleep_duration_hours', data=df, scatter_kws={'s': 10})
plt.title('Caffeine Intake vs Sleep Duration')
plt.xlabel('Caffeine Intake (mg/day)')
plt.ylabel('Sleep Duration (hours)')
plt.show()

# %% Countplot: Location Type
df['location_type'] = df[['location_type_rural', 'location_type_suburban', 'location_type_urban']].idxmax(axis=1)
df['location_type'] = df['location_type'].str.replace('location_type_', '')

plt.figure(figsize=(6, 4))
sns.countplot(x='location_type', data=df)
plt.title('Participant Count by Location Type')
plt.xlabel('Location Type')
plt.ylabel('Count')
plt.show()

# %%
