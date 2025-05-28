# 결측치를 임의로 삽입하는 코드입니다. (원본 데이터에 결측치가 없으므로)

import numpy as np
import pandas as pd

# 원본 데이터 불러오기
df = pd.read_csv("digital_diet_mental_health.csv")

# 결측치 삽입 함수 정의
def insert_missing_values(df, column, frac=0.05, random_state=42):
    np.random.seed(random_state)
    missing_indices = df.sample(frac=frac).index
    df.loc[missing_indices, column] = np.nan
    return df

df_missing = df.copy()

# 선택한 열에 결측치 삽입
df_missing = insert_missing_values(df_missing, "age", frac=0.05)
df_missing = insert_missing_values(df_missing, "daily_screen_time_hours", frac=0.05)
df_missing = insert_missing_values(df_missing, "sleep_duration_hours", frac=0.05)
df_missing = insert_missing_values(df_missing, "location_type", frac=0.05)

# 결과 확인
print(df_missing.isna().sum())

# 새 파일로 저장
df_missing.to_csv("digital_diet_mental_health_with_missing.csv", index=False)
