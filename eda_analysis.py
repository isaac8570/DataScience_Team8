#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("preprocessed_dataset.csv")  # 같은 폴더에 있는지 확인
df.head()
df.info()
df.describe()

# %% 스크린타임 분포
sns.histplot(df['num__daily_screen_time_hours'], kde=True)
plt.title("Daily Screen Time Distribution")
plt.xlabel("Hours per Day (Standardized)")
plt.ylabel("Count")
plt.show()

# %% 수면시간 vs 스트레스 (박스플롯)
plt.figure(figsize=(10, 6))
sns.boxplot(x='num__stress_level', y='num__sleep_duration_hours', data=df)
plt.title("Sleep Duration by Stress Level")
plt.xlabel("Stress Level (Standardized)")
plt.ylabel("Sleep Duration (Standardized)")
plt.xticks(rotation=45)  # 👈 x축 레이블 겹침 방지
plt.tight_layout()
plt.show()

# %% 전체 상관관계 히트맵
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# %%  소셜미디어 사용 vs 정신건강 점수 (산점도)
sns.scatterplot(x='num__social_media_hours', y='num__mental_health_score', data=df)
plt.title("Mental Health vs Social Media Usage")
plt.xlabel("Social Media Hours")
plt.ylabel("Mental Health Score")
plt.show()
# %%
