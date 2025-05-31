import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv("preprocessed_dataset.csv")


'''
    Part 1. 3가지 변수(sleep_efficiency, mental_health_index, wellness_score)로 각각 total_screen_time에 대해
        Linear Regression과 Random Forest Regression을 진행함 
        Cross Validation - neg_mean_absolute_error, neg_mean_squared_error, r2을 통해 model의 성능을 평가함
        추가적으로 각 모델의 r2지표를 비교하여 변수 별로 어떤 Regression이 적합했는지 판단함
'''

# 개별 변수 분석을 위한 함수 정의 : sleep_efficiency, mental_health_index, wellness_score을 각각 분석하기 위한 함수
def analyze_single_variable(X_var, y_var, var_name, data):
    """단일 변수에 대한 회귀분석 수행"""
    print(f"\n{'='*60}")
    print(f" Feature Analysis < {var_name.upper()} >")
    print(f"{'='*60}")
    
    # feature와 target value 설정
    X = data[[X_var]].values
    y = data[y_var].values
    
    # train/test 데이터로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
    # Regression 방정식 출력
    print(f"\nLinear Regression :")
    print(f"total_screen_time = {lr_model.intercept_:.2f} + ({lr_model.coef_[0]:.2f}) × {X_var}")
    
    
    # 2. Random Forest Regression model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    # 3. K-Fold Cross Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Cross Validation 점수 계산 
    # 1. Linear Regression 
    lr_cv_mae = -cross_val_score(lr_model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    lr_cv_rmse = np.sqrt(-cross_val_score(lr_model, X, y, cv=kfold, scoring='neg_mean_squared_error'))
    lr_cv_r2 = cross_val_score(lr_model, X, y, cv=kfold, scoring='r2')
    
    # 2. Random Forest Regression
    rf_cv_mae = -cross_val_score(rf_model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    rf_cv_rmse = np.sqrt(-cross_val_score(rf_model, X, y, cv=kfold, scoring='neg_mean_squared_error'))
    rf_cv_r2 = cross_val_score(rf_model, X, y, cv=kfold, scoring='r2')
    
    # 4. 모델 성능 지표 계산
    # 1. Linear Regression
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_r2 = r2_score(y_test, y_pred_lr)
    
    # 2. Random Forest Regression
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_r2 = r2_score(y_test, y_pred_rf)
    
    # Correlation 계산 - [ X 변수 <-> total_screen_time ]
    correlation = data[X_var].corr(data[y_var])
    
    # 결과 출력
    print(f"\n1. Correlation: {correlation:.4f}")
    
    print(f"\n < Linear Regression >")
    print(f"   - MAE: {lr_mae:.2f}, RMSE: {lr_rmse:.2f}, R²: {lr_r2:.4f}")
    print(f"   - CV MAE: {lr_cv_mae.mean():.2f} (±{lr_cv_mae.std():.2f})")
    print(f"   - CV RMSE: {lr_cv_rmse.mean():.2f} (±{lr_cv_rmse.std():.2f})")
    print(f"   - CV R²: {lr_cv_r2.mean():.4f} (±{lr_cv_r2.std():.4f})")
    
    print(f"\n < Random Forest >")
    print(f"   - MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}, R²: {rf_r2:.4f}")
    print(f"   - CV MAE: {rf_cv_mae.mean():.2f} (±{rf_cv_mae.std():.2f})")
    print(f"   - CV RMSE: {rf_cv_rmse.mean():.2f} (±{rf_cv_rmse.std():.2f})")
    print(f"   - CV R²: {rf_cv_r2.mean():.4f} (±{rf_cv_r2.std():.4f})")
    
    # 각 변수 별 더 나은 모델 선택 ; LInear Regression vs Random Forest
        # 각 model의 r2 지표를 활용하여 판단함
    better_model = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
    print(f"\n2. Better Model : {better_model}")
    
    # analyze_single_variable 함수의 Return value
    return {
        'variable': X_var,
        'correlation': correlation,
        'lr_mae': lr_mae, 'lr_rmse': lr_rmse, 'lr_r2': lr_r2,
        'rf_mae': rf_mae, 'rf_rmse': rf_rmse, 'rf_r2': rf_r2,
        'lr_cv_mae': lr_cv_mae.mean(), 'rf_cv_mae': rf_cv_mae.mean(),
        'X_test': X_test, 'y_test': y_test, 'y_pred_lr': y_pred_lr, 'y_pred_rf': y_pred_rf,
        'lr_model': lr_model, 'rf_model': rf_model,
        'X': X, 'y': y
    }


# 각 변수별 개별 분석을 진행 - analyze_single_variable를 활용
results = {}

# 1. Sleep Efficiency
results['sleep'] = analyze_single_variable('sleep_efficiency', 'total_screen_time', 'Sleep Efficiency', data)

# 2. Mental Health Index
results['mental'] = analyze_single_variable('mental_health_index', 'total_screen_time', 'Mental Health Index', data)

# 3. Wellness Score
results['wellness'] = analyze_single_variable('wellness_score', 'total_screen_time', 'Wellness Score', data)


# 3가지를 종합 했을 때의 비교 및 분석 진행
print(f"\n{'='*60}")
print("Feature별 성능 비교")
print(f"{'='*60}")

    # 비교를 위한 새로운 Data frame 생성 - 각 변수별 개별 분석을 진행한 result 값을 활용함
comparison_df = pd.DataFrame({
    'Variable': ['Sleep Efficiency', 'Mental Health Index', 'Wellness Score'],
    'Correlation': [results['sleep']['correlation'], results['mental']['correlation'], results['wellness']['correlation']],
    'LR_R2': [results['sleep']['lr_r2'], results['mental']['lr_r2'], results['wellness']['lr_r2']],
    'RF_R2': [results['sleep']['rf_r2'], results['mental']['rf_r2'], results['wellness']['rf_r2']],
    'LR_MAE': [results['sleep']['lr_mae'], results['mental']['lr_mae'], results['wellness']['lr_mae']],
    'RF_MAE': [results['sleep']['rf_mae'], results['mental']['rf_mae'], results['wellness']['rf_mae']]
})
print(comparison_df.round(4))


# 시각화
fig = plt.figure(figsize=(15, 10))

# 1. 개별 변수 산점도 및 회귀선 (3x2 grid)
variables = ['sleep_efficiency', 'mental_health_index', 'wellness_score'] # Variable
var_names = ['Sleep Efficiency', 'Mental Health Index', 'Wellness Score'] # 그래프에 나타내기 위한 Variable name
colors = ['slategrey', 'lightsteelblue', 'cornflowerblue'] 

for i, (var, name, color) in enumerate(zip(variables, var_names, colors)):
    # Scatter Plot
    plt.subplot(3, 4, i*4 + 1)
    plt.scatter(data[var], data['total_screen_time'], alpha=0.6, color=color)
    
    # Linear Regression
    z = np.polyfit(data[var], data['total_screen_time'], 1)
    p = np.poly1d(z)
    plt.plot(data[var], p(data[var]), "r--", alpha=0.8, linewidth=2)
    
    corr = data[var].corr(data['total_screen_time'])
    plt.xlabel(name)
    plt.ylabel('Total Screen Time (minutes)')
    plt.title(f'{name} vs Screen Time\nCorrelation: {corr:.3f}')
    plt.grid(True, alpha=0.3)


# 2. 각 변수별 예측 성능 비교 (실제 vs predict) - model
for i, (key, name) in enumerate(zip(['sleep', 'mental', 'wellness'], var_names)):
    result = results[key]
    
    # Linear Regression
    plt.subplot(3, 4, i*4 + 2)
    plt.scatter(result['y_test'], result['y_pred_lr'], alpha=0.6, color='blue')
    plt.plot([result['y_test'].min(), result['y_test'].max()], 
             [result['y_test'].min(), result['y_test'].max()], 'r--', lw=2)
    plt.xlabel('Actual Screen Time')
    plt.ylabel('Predicted Screen Time')
    plt.title(f'{name}\nLinear Regression (R² = {result["lr_r2"]:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Random Forest
    plt.subplot(3, 4, i*4 + 3)
    plt.scatter(result['y_test'], result['y_pred_rf'], alpha=0.6, color='green')
    plt.plot([result['y_test'].min(), result['y_test'].max()], 
             [result['y_test'].min(), result['y_test'].max()], 'r--', lw=2)
    plt.xlabel('Actual Screen Time')
    plt.ylabel('Predicted Screen Time')
    plt.title(f'{name}\nRandom Forest (R² = {result["rf_r2"]:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Linear Regression + Random Forest
    plt.subplot(3, 4, i*4 + 4)
    residuals_lr = result['y_test'].flatten() - result['y_pred_lr']
    residuals_rf = result['y_test'].flatten() - result['y_pred_rf']
    
    plt.scatter(result['y_pred_lr'], residuals_lr, alpha=0.6, color='blue', label='Linear Reg')
    plt.scatter(result['y_pred_rf'], residuals_rf, alpha=0.6, color='green', label='Random Forest')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{name}\nResidual Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



# 성능 비교 시각화 - 막대 그래프
fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

# 1. Correlation 비교
variables_full = ['Sleep Efficiency', 'Mental Health Index', 'Wellness Score']
correlations = [results['sleep']['correlation'], results['mental']['correlation'], results['wellness']['correlation']]
colors_bar = ['skyblue', 'lightgreen', 'lightcoral']

axes[0,0].bar(variables_full, correlations, color=colors_bar)
axes[0,0].set_title('Correlation with Screen Time')
axes[0,0].set_ylabel('Correlation Coefficient')
axes[0,0].tick_params(axis='x', rotation=15)
for i, v in enumerate(correlations): # github 코드 참고
    axes[0,0].text(i, v / 2, f'{v:.3f}', ha='center', va='center', fontsize=10, color='black')
# 2. Linear Regression R^2 비교
lr_r2_scores = [results['sleep']['lr_r2'], results['mental']['lr_r2'], results['wellness']['lr_r2']]
axes[0,1].bar(variables_full, lr_r2_scores, color=colors_bar)
axes[0,1].set_title('Linear Regression R^2 Score')
axes[0,1].set_ylabel('R^2 Score')
axes[0,1].tick_params(axis='x', rotation=15)
for i, v in enumerate(lr_r2_scores): # github 코드 참고
    axes[0,1].text(i, v / 2, f'{v:.3f}', ha='center', va='center', fontsize=10, color='black')

# 3. Random Forest R^2 비교
rf_r2_scores = [results['sleep']['rf_r2'], results['mental']['rf_r2'], results['wellness']['rf_r2']]
axes[0,2].bar(variables_full, rf_r2_scores, color=colors_bar)
axes[0,2].set_title('Random Forest R^2 Score')
axes[0,2].set_ylabel('R^2 Score')
axes[0,2].tick_params(axis='x', rotation=15)
for i, v in enumerate(rf_r2_scores): # github 코드 참고
    axes[0,2].text(i, v / 2, f'{v:.3f}', ha='center', va='center', fontsize=10, color='black')

# 4. MAE 비교
lr_mae_scores = [results['sleep']['lr_mae'], results['mental']['lr_mae'], results['wellness']['lr_mae']]
rf_mae_scores = [results['sleep']['rf_mae'], results['mental']['rf_mae'], results['wellness']['rf_mae']]

x_pos = np.arange(len(variables_full))
width = 0.5

axes[1,0].bar(x_pos - width/2, lr_mae_scores, width, label='Linear Regression', color='skyblue')
axes[1,0].bar(x_pos + width/2, rf_mae_scores, width, label='Random Forest', color='lightgreen')
axes[1,0].set_title('Mean Absolute Error (MAE)')
axes[1,0].set_ylabel('MAE')
axes[1,0].set_xticks(x_pos)
axes[1,0].tick_params(axis='x', rotation=15)
axes[1,0].set_xticklabels(variables_full)
axes[1,0].legend(loc='lower right')

# 5. RMSE 비교
lr_rmse_scores = [results['sleep']['lr_rmse'], results['mental']['lr_rmse'], results['wellness']['lr_rmse']]
rf_rmse_scores = [results['sleep']['rf_rmse'], results['mental']['rf_rmse'], results['wellness']['rf_rmse']]

axes[1,1].bar(x_pos - width/2, lr_rmse_scores, width, label='Linear Regression', color='skyblue')
axes[1,1].bar(x_pos + width/2, rf_rmse_scores, width, label='Random Forest', color='lightgreen')
axes[1,1].set_title('Root Mean Square Error (RMSE)')
axes[1,1].set_ylabel('RMSE')
axes[1,1].set_xticks(x_pos)
axes[1,1].tick_params(axis='x', rotation=15)
axes[1,1].set_xticklabels(variables_full)
axes[1,1].legend(loc='lower right')

# 6. Cross-validation MAE 비교
lr_cv_mae_scores = [results['sleep']['lr_cv_mae'], results['mental']['lr_cv_mae'], results['wellness']['lr_cv_mae']]
rf_cv_mae_scores = [results['sleep']['rf_cv_mae'], results['mental']['rf_cv_mae'], results['wellness']['rf_cv_mae']]

axes[1,2].bar(x_pos - width/2, lr_cv_mae_scores, width, label='Linear Regression', color='skyblue')
axes[1,2].bar(x_pos + width/2, rf_cv_mae_scores, width, label='Random Forest', color='lightgreen')
axes[1,2].set_title('Cross-Validation MAE')
axes[1,2].set_ylabel('CV MAE')
axes[1,2].set_xticks(x_pos)
axes[1,2].tick_params(axis='x', rotation=15)
axes[1,2].set_xticklabels(variables_full)
axes[1,2].legend(loc='lower right')

plt.show()


# 최종 결론
print(f"\n{'='*60}")
print("< Final Analysis >")
print(f"{'='*60}")

# Correlation 순위
corr_ranking = sorted(zip(variables_full, correlations), key=lambda x: abs(x[1]), reverse=True)
print(f"\n < Correlation 순위 >") # 절대값으로 사용 
for i, (var, corr) in enumerate(corr_ranking, 1):
    print(f"   {i}. {var}: {corr:.4f}")

# Linear Regression R^2 순위
lr_ranking = sorted(zip(variables_full, lr_r2_scores), key=lambda x: x[1], reverse=True)
print(f"\n < Linear Regression R^2 >")
for i, (var, r2) in enumerate(lr_ranking, 1):
    print(f"   {i}. {var}: {r2:.4f}")

# Random Forest R^2 순위
rf_ranking = sorted(zip(variables_full, rf_r2_scores), key=lambda x: x[1], reverse=True)
print(f"\n < Random Forest R^2 >")
for i, (var, r2) in enumerate(rf_ranking, 1):
    print(f"   {i}. {var}: {r2:.4f}")

print(f"\n< Insight >")
best_var = corr_ranking[0][0]
best_corr = corr_ranking[0][1]
print(f"   - 스크린타임 예측에 가장 영향력 있는 변수 : {best_var}")
print(f"   - Correlation 정도 : {best_corr:.4f}")

# 개별 회귀 방정식 요약
print(f"\n< Variable Regression 방정식 >")
for key, name in zip(['sleep', 'mental', 'wellness'], variables_full):
    model = results[key]['lr_model']
    print(f"   • {name}:")
    print(f"     total_screen_time = {model.intercept_:.2f} + ({model.coef_[0]:.2f}) × {key.replace('_', ' ')}")
