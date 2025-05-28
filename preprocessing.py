import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


df = pd.read_csv("digital_diet_mental_health_with_missing.csv")

# 파생 변수 생성
## 1. 스크린 타임 관련
df['total_screen_time'] = df['phone_usage_hours'] + df['laptop_usage_hours'] + \
                          df['tablet_usage_hours'] + df['tv_usage_hours']

df['digital_entertainment_time'] = df['social_media_hours'] + \
                                 df['entertainment_hours'] + df['gaming_hours']

## 2. 건강 관련
df['sleep_efficiency'] = df['sleep_quality'] / df['sleep_duration_hours'].replace(0, np.nan)

df['mental_health_index'] = (df['mental_health_score'] - df['weekly_anxiety_score'] - \
                           df['weekly_depression_score']) / 3

df['wellness_score'] = (df['sleep_quality'] + df['mood_rating'] + \
                       df['mental_health_score']) / 3

## 3. 생활 패턴 관련
df['digital_wellness_balance'] = df['mindfulness_minutes_per_day'] / \
        (df['total_screen_time'].replace(0, np.nan) * 60)


# 기존 컬럼 삭제
df = df.drop(columns=['user_id'])

# 결측치 확인
print("결측치 개수:")
print(df.isna().sum())

# 결측치 비율
print("\n결측치 비율:")
print((df.isna().sum() / len(df) * 100).round(2))

time_cols = ['daily_screen_time_hours', 'phone_usage_hours', 
            'laptop_usage_hours', 'tablet_usage_hours', 'tv_usage_hours', 
            'social_media_hours', 'work_related_hours', 'entertainment_hours', 
            'gaming_hours', 'sleep_duration_hours', 'physical_activity_hours_per_week',
            'total_screen_time', 'digital_entertainment_time']

score_cols = ['sleep_quality', 'mood_rating', 'stress_level',
             'mental_health_score', 'weekly_anxiety_score', 'weekly_depression_score',
             'sleep_efficiency', 'mental_health_index', 'wellness_score']

ratio_cols = ['digital_wellness_balance']

binary_cols = ['uses_wellness_apps', 'eats_healthy']

special_cols = ['age', 'caffeine_intake_mg_per_day']

cat_cols = ['gender', 'location_type']

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method='iqr', threshold=1.5):
        self.method = method
        self.threshold = threshold
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.method == 'iqr':
            # NaN 값을 제외하고 계산
            Q1 = np.nanpercentile(X, 25, axis=0)
            Q3 = np.nanpercentile(X, 75, axis=0)
            IQR = Q3 - Q1
            self.lower_bounds_ = Q1 - self.threshold * IQR
            self.upper_bounds_ = Q3 + self.threshold * IQR
        elif self.method == 'zscore':
            # NaN 값을 제외하고 계산
            mean = np.nanmean(X, axis=0)
            std = np.nanstd(X, axis=0)
            self.lower_bounds_ = mean - self.threshold * std
            self.upper_bounds_ = mean + self.threshold * std
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            # DataFrame을 numpy array로 변환
            X_array = X.values.copy()
            # 각 컬럼별로 클리핑 적용
            for i in range(X_array.shape[1]):
                mask = ~np.isnan(X_array[:, i])
                X_array[mask, i] = np.clip(X_array[mask, i],
                                         self.lower_bounds_[i],
                                         self.upper_bounds_[i])
            # 다시 DataFrame으로 변환
            X = pd.DataFrame(X_array, columns=X.columns, index=X.index)
        else:
            X = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X
        
    def get_feature_names_out(self, feature_names_in=None):
        if feature_names_in is None:
            return None
        return feature_names_in

# 이상치 처리만 먼저 수행
df_outlier_handled = df.copy()

# 각 변수 그룹별 이상치 처리
for cols, method, threshold in [
    (time_cols, 'iqr', 1.5),
    (score_cols, 'zscore', 3.0),
    (ratio_cols, 'iqr', 2.0),
    (special_cols, 'zscore', 3.0)
]:
    handler = OutlierHandler(method=method, threshold=threshold)
    # DataFrame 형태로 변환하여 처리
    temp_df = pd.DataFrame(df[cols])
    df_outlier_handled[cols] = handler.fit_transform(temp_df)

# 이상치 처리 전후 통계량 비교
print("\n=== 이상치 처리 전후 통계량 비교 ===")
print("\n처리 전:")
print(df[time_cols + score_cols + ratio_cols + special_cols].describe())

print("\n이상치 처리 후 (스케일링 전):")
print(df_outlier_handled[time_cols + score_cols + ratio_cols + special_cols].describe())

# 변수 타입별 파이프라인 (이상치 처리 제외)
## 1) 시간 관련 변수
time_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())  
])

## 2) 점수/등급 관련 변수
score_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

## 3) 비율 관련 변수
ratio_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

## 4) 이진 변수
binary_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

## 5) 특수 변수
special_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

## 6) 범주형 변수
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# 컬럼별 전처리 조합
preprocessor = ColumnTransformer([
    ("screen_time", time_pipe, time_cols),
    ("health_score", score_pipe, score_cols),
    ("balance_ratio", ratio_pipe, ratio_cols),
    ("binary_feature", binary_pipe, binary_cols),
    ("demographic", special_pipe, special_cols),
    ("category", cat_pipe, cat_cols)
])

# 이상치 처리된 데이터에 대해 스케일링 적용
X_ready = preprocessor.fit_transform(df_outlier_handled)
feature_names = preprocessor.get_feature_names_out()

# 컬럼 이름 정리
clean_feature_names = []
for name in feature_names:
    clean_name = name.split('__')[-1]
    clean_name = clean_name.lower().replace(' ', '_')
    clean_feature_names.append(clean_name)

# 전처리된 데이터프레임 생성
df_ready = pd.DataFrame(X_ready, columns=clean_feature_names)

print("\n=== 이상치 처리 전후 통계량 비교 ===")
print("\n처리 전:")
print(df[time_cols + score_cols + ratio_cols + special_cols].describe())

print("\n이상치 처리 후 (스케일링 전):")
print(df_outlier_handled[time_cols + score_cols + ratio_cols + special_cols].describe())

print("\n최종 전처리 후 (이상치 처리 + 스케일링):")
print(df_ready.describe())

# 파일 저장 시도 전에 기존 파일이 열려있는지 확인
try:
    df_ready.to_csv("preprocessed_dataset.csv", index=False)
    df_outlier_handled.to_csv("outlier_handled_dataset.csv", index=False)
except PermissionError:
    print("\n파일 저장 권한 오류가 발생했습니다.")
    print("다음 파일들이 다른 프로그램에서 열려있는지 확인해주세요:")
    print("1. preprocessed_dataset.csv")
    print("2. outlier_handled_dataset.csv")
    print("\n파일을 닫고 다시 실행해주세요.")