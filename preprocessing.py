import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, MinMaxScaler,
    OrdinalEncoder
)
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

print(df[columns].isna().sum())  # -> there are no NaN value

# mindfulness_minutes_per_day 컬럼만 minute 단위, 나머지는 다 hour 단위
# -> 전처리 과정에서 시간 단위로 바꿀까 했지만, 일단 분 단위가 더 반영이 잘 될듯 하여 해당 부분은 보류해 두었습니다.
