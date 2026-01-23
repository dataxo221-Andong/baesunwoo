
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

import joblib
from sklearn.cluster import KMeans
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pkl_path = 'models/ft_100.pkl'

if not os.path.exists(pkl_path):
    print(f"[오류]: '{pkl_path}' 파일이 없습니다. 현재 폴더에 파일을 넣어주세요.")
    exit()

# ... (replacement logic handled by multiple chunks below)
df = pd.read_pickle(pkl_path)

# 1) 불량 라벨이 없는 데이터 제거
if 'waferIndex' in df.columns:
    df = df.drop(['waferIndex'], axis=1)

# 9개 타겟 라벨 필터링
target_labels = ['CENTER', 'DONUT', 'EDGE-LOC', 'EDGE-RING', 'LOC', 'NEAR-FULL', 'RANDOM', 'SCRATCH', 'NONE']
df = df[df['failureType'].isin(target_labels)].reset_index(drop=True)

print(f"[완료]: 전처리 완료: 총 {len(df)}장의 유효한 웨이퍼 데이터 확보")

samples_per_class = 100  # 클래스별 ~장씩만 샘플링
balanced_df = pd.DataFrame()

print(f"[진행]: 데이터 샘플링 진행 (유형별 {samples_per_class}장)...")

for label in target_labels:
    label_df = df[df['failureType'] == label]
    if len(label_df) > 0:
        n_samples = min(len(label_df), samples_per_class)
        sampled = label_df.sample(n=n_samples, replace=False, random_state=42)
        balanced_df = pd.concat([balanced_df, sampled])

df = balanced_df.reset_index(drop=True)
print(f"[완료]: 최종 분석 대상 데이터: {len(df)}장 (샘플링 완료)")

# ==========================================
# 2. 데이터셋 클래스 정의 (제공해주신 코드 그대로 사용)
# ==========================================
class WaferDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.label_map = {label: i for i, label in enumerate(target_labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wafer_map = self.data.iloc[idx]['waferMap']
        
        # 64x64 리사이징 (학습 코드와 동일)
        wafer_map = cv2.resize(wafer_map.astype('float32'), (64, 64), interpolation=cv2.INTER_NEAREST)
        
        # 정규화 (/ 2.0)
        wafer_map = wafer_map / 2.0
        
        # 차원 추가 (1채널)
        wafer_tensor = torch.tensor(wafer_map, dtype=torch.float32).unsqueeze(0)

        # 라벨 인덱싱
        label_str = self.data.iloc[idx]['failureType']
        label_idx = self.label_map[label_str]

        return wafer_tensor, label_idx

# 데이터셋 인스턴스 생성
full_dataset = WaferDataset(df)

# DataLoader 생성 (특징 추출용이므로 셔플 X)
batch_size = 64
train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
print(f"[완료]: 데이터셋 준비 완료: 총 {len(full_dataset)}장")

# ==========================================
# 3. 모델 아키텍처 정의 및 가중치 로드
# ==========================================
print("[진행]: 모델 구조 생성 중...")
# 1) ResNet18 구조 생성 (가중치를 담기 위한 뼈대)
model = models.resnet18(pretrained=False) # 이미 학습된 파일을 쓸 것이므로 백지 상태로 시작

# 2) 모델 구조 수정 (1채널 입력, 9개 출력 - 학습시 설정과 동일해야 함)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 9)
    
# 3) 저장된 가중치(wafer_classifier.pth) 로드
model_path = 'models/wafer_classifier_10.pth'
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[완료]: 학습된 모델 파일 로드 성공: {model_path}")
    except Exception as e:
        print(f"[오류]: 가중치 로드 중 오류 발생: {e}")
        exit()
else:
    print(f"[오류]: '{model_path}' 파일이 없습니다.")
    exit()

# 4) 특징 추출기로 변환 (마지막 분류 레이어 제거)
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.to(device)
feature_extractor.eval()

# ==========================================
# 4. 특징 벡터 추출 (Inference)
# ==========================================
print("[진행]: 전체 데이터 특징(Feature) 추출 시작...")
features_list = []
labels_list = []

with torch.no_grad():
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        
        # 모델 통과
        outputs = feature_extractor(inputs)
        
        # (Batch, 512, 1, 1) -> (Batch, 512) 1차원 벡터로 변환
        outputs = outputs.view(outputs.size(0), -1)
        
        features_list.append(outputs.cpu().numpy())
        labels_list.append(labels.numpy())
        
        if (i+1) % 100 == 0:
            print(f"   - {i+1}번째 배치 처리 완료...")

# 리스트 병합
X_features = np.concatenate(features_list, axis=0)
true_labels = np.concatenate(labels_list, axis=0)
print(f"[완료]: 특징 추출 완료. 데이터 형태: {X_features.shape}")

# 데이터 정규화 (StandardScaler 적용)
# 특징값들의 평균을 0, 분산을 1로 맞춰서 K-Means가 거리를 더 잘 계산하게 함
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)
print("[진행]: 특징 벡터 스케일링(StandardScaler) 완료")

# ==========================================
# 5. K-Means 군집화 수행
# ==========================================
# 클러스터 개수 설정 (불량 유형 개수인 9개로 시도)
n_clusters = 9
print(f"[진행]: K-Means 클러스터링 시작 (k={n_clusters})...")

# n_init=50 유지 (안정적인 결과)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
kmeans.fit(X_features)

print("[완료]: 군집화 완료!")

# 모델 저장
joblib.dump(kmeans, 'models/kmeans_model.pkl')
print("  [저장]: K-Means 모델 저장 완료: kmeans_model.pkl")

# ==========================================
# 6. 결과 분석 (클러스터별 주성분)
# ==========================================
print("\n[결과]: 클러스터 분석 결과:")

# 숫자 라벨 -> 문자열 맵핑 (출력용)
idx_to_class = {v: k for k, v in full_dataset.label_map.items()}
cluster_labels = kmeans.labels_

for i in range(n_clusters):
    indices = np.where(cluster_labels == i)[0]
    
    if len(indices) > 0:
        curr_true_labels = true_labels[indices]
        
        # 가장 많이 나온 라벨 찾기
        unique, counts = np.unique(curr_true_labels, return_counts=True)
        top_idx = np.argsort(counts)[-1]
        
        major_label_idx = unique[top_idx]
        major_label_name = idx_to_class[major_label_idx]
        major_count = counts[top_idx]
        ratio = (major_count / len(indices)) * 100
        
        print(f"[Cluster {i}] 할당 {len(indices)}개 -> 최빈값: '{major_label_name}' ({ratio:.1f}%)")
    else:
        print(f"[Cluster {i}] 데이터 없음")

# ==========================================
# 7. 결과 시각화 (PCA 2차원 축소)
# ==========================================
print("[진행]: 결과 그래프 시각화(PCA) 진행 중...")

# 벡터 -> 2차원으로 압축
pca = PCA(n_components=2)
pca_features = pca.fit_transform(X_features)

plt.figure(figsize=(10, 8))
# 클러스터별로 색을 다르게 해서 산점도 그리기
for i in range(n_clusters):
    # 해당 클러스터 데이터만 골라내기
    cluster_points = pca_features[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}', alpha=0.6, s=10)

plt.title(f'K-Means Clustering Result (k={n_clusters})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()