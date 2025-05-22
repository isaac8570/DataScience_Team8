import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler)
from sklearn.impute import SimpleImputer


df = pd.read_csv("digital_diet_mental_health.csv")

columns = ['user_id', 'age', 'gender', 'daily_screen_time_hours',
       'phone_usage_hours', 'laptop_usage_hours', 'tablet_usage_hours',
       'tv_usage_hours', 'social_media_hours', 'work_related_hours',
       'entertainment_hours', 'gaming_hours', 'sleep_duration_hours',
       'sleep_quality', 'mood_rating', 'stress_level',
       'physical_activity_hours_per_week', 'location_type',
       'mental_health_score', 'uses_wellness_apps', 'eats_healthy',
       'caffeine_intake_mg_per_day', 'weekly_anxiety_score',
       'weekly_depression_score', 'mindfulness_minutes_per_day']

df = df.drop(columns=['user_id']) # 사용되지 않는 칼럼 삭제

# print(df.isna().sum())  # -> there are no NaN value

# mindfulness_minutes_per_day 컬럼만 minute 단위, 나머지는 다 hour 단위
# -> 전처리 과정에서 시간 단위로 바꿀까 했지만, 일단 분 단위가 더 반영이 잘 될듯 하여 해당 부분은 보류해 두었습니다.

# df['mindfulness_hours'] = df['mindfulness_minutes_per_day'] / 60
# num_cols.append('mindfulness_hours')
# num_cols.remove('mindfulness_minutes_per_day')

# 수치형, 범주형 변수 리스트

num_cols = ['age', 'daily_screen_time_hours', 'phone_usage_hours', 
            'laptop_usage_hours', 'tablet_usage_hours', 'tv_usage_hours', 
            'social_media_hours', 'work_related_hours', 'entertainment_hours', 
            'gaming_hours', 'sleep_duration_hours','sleep_quality', 'mood_rating', 
            'stress_level','physical_activity_hours_per_week','mental_health_score', 
            'uses_wellness_apps', 'eats_healthy', 'caffeine_intake_mg_per_day', 
            'weekly_anxiety_score','weekly_depression_score', 'mindfulness_minutes_per_day',
       ]

cat_cols = ['gender','location_type']

# pipeline
##  1) 수치형
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # 결측값 -> 중앙값으로 넣기 (결측값 없지만 일단 코드에 넣었었음)
    ("scaler", StandardScaler())                     # 표준화 - StandardScaler
])

 ## 2) 범주형
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # 결측값 -> 최빈값
    ("onehot", OneHotEncoder(handle_unknown="ignore"))     # one hot encoding
])

# 컬럼별 전처리 조합
preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# 전체 데이터 전처리 실행
X_ready = preprocessor.fit_transform(df)

# 열이름 그대로 가져오기
feature_names = preprocessor.get_feature_names_out()

# df_ready(전처리 완료된 df) DataFrame으로 변환
df_ready = pd.DataFrame(X_ready, columns=feature_names)

# print(df_ready.head())

df_ready.to_csv("preprocessed_dataset.csv", index=False)