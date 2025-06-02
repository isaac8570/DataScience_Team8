import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 데이터 로드
data = pd.read_csv("preprocessed_dataset.csv")

# 사용할 feature 정의
features = ['total_screen_time', 'sleep_efficiency', 'mental_health_index', 'wellness_score']
X = data[features].dropna()

# Standardzation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# k=2,3,4 각각에 대해 KMeans 수행 및 시각화
cluster_results = {}

# Kmeans 활용하기
for k in [2, 3, 4]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 클러스터 결과 저장
    data[f'Cluster_k{k}'] = clusters
    
    # 라벨링 자동화
    data[f'Cluster_Label_k{k}'] = data[f'Cluster_k{k}'].apply(lambda x: f"Group {chr(65 + x)}")
    
    # 평균 특성 요약
    summary = data.groupby(f'Cluster_Label_k{k}')[features].mean().round(3)
    cluster_results[k] = {
        'summary': summary,
        'labels': data[f'Cluster_Label_k{k}']
    }

    # Silhouette Score 계산
    silhouette = silhouette_score(X_scaled, clusters)
    cluster_results[k]['silhouette'] = silhouette

    # PCA 시각화
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7, 5))
    for label in data[f'Cluster_Label_k{k}'].unique():
        idx = data[f'Cluster_Label_k{k}'] == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, alpha=0.7)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title(f"K={k} Clustering based on Mental Health")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\n [K={k}] Cluster별 평균 특성 요약 :")
    print(summary)
    print(f"K={k} Silhouette Score: {silhouette:.4f}")

# Silhouette Score 시각화
k_values = list(cluster_results.keys())
silhouette_scores = [cluster_results[k]['silhouette'] for k in k_values]

plt.figure(figsize=(6,4))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for Different K")
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
    
    
# Elbow Method (최적 k 자동 추정) ----------- Github 참고해서 추가 작업 진행해봤습니다 ; 최적의 k 찾기
inertia = []
silhouette_scores = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    clusters = kmeans.predict(X_scaled)
    inertia.append(kmeans.inertia_)

    # silhouette score는 k=1에서 계산이 안됨
    if(k != 1):
        silhouette = silhouette_score(X_scaled, clusters)
        silhouette_scores.append(silhouette)
    else:
        silhouette_scores.append(np.nan)

# Elbow 지점 자동 계산
inertia_diff = np.diff(inertia)
inertia_diff2 = np.diff(inertia_diff)
elbow_k = np.argmax(inertia_diff2) + 2
print(f"< Elbow Method optimal k: {elbow_k} >")

# Elbow Plot
plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, 'bo-')
plt.axvline(x=elbow_k, color='red', linestyle='--', label=f'Elbow k={elbow_k}')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Silhouette Score Plot
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'go-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by k')
plt.xticks(K_range)
plt.grid(True)
plt.tight_layout()
plt.show()