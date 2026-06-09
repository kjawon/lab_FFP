import os
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from google.colab import drive
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, classification_report
# [핵심 변경] HDBSCAN을 내리고, K-Means를 다시 소환하오!
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import torch.nn.functional as F

# ==========================================
# 1. 구글 드라이브 마운트 및 경로 설정
# ==========================================
print("도로로 병장: 요새(드라이브)를 연결하고 ArcFace 모델을 찾고 있소이다!")
drive.mount('/content/drive')

# ArcFace와 투트랙으로 훈련한 최강의 모델 경로!
model_save_path = '/content/drive/MyDrive/Forensic_AI_Ultimate/ArcFace_TwoStage_Deepzzle.pth'

# ==========================================
# 2. 데이터 재조달 (검증용 병력 소집)
# ==========================================
print("\n도로로 병장: 검증을 위해 실전 데이터를 AWS S3에서 다시 조달하오!")
base_url = "https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/"
raw_data_dir = "/content/govdocs_all_classes"
os.makedirs(raw_data_dir, exist_ok=True)

zip_ids = [f"{i:03d}" for i in range(5)]
for zid in zip_ids:
    zip_path = f"/content/{zid}.zip"
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(base_url + f"{zid}.zip", zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_data_dir)
    if os.path.exists(zip_path): os.remove(zip_path)

# ==========================================
# 3. 데이터 전처리
# ==========================================
def calc_entropy_map(byte_array, block_size=64):
    reshaped = byte_array.reshape(-1, block_size)
    ents = []
    for row in reshaped:
        counts = np.bincount(row, minlength=256)
        p = counts[counts > 0] / block_size
        ents.append(-np.sum(p * np.log2(p)))
    return np.array(ents, dtype=np.float32) / 8.0

class RealWorldGovDocsDataset(Dataset):
    def __init__(self, data_dir, chunk_size=4096, min_class_samples=100):
        self.chunk_size = chunk_size
        self.fragments, self.histograms, self.entropy_maps = [], [], []
        self.labels, self.file_ids = [], []
        self.ext_to_label, self.label_to_ext = {}, {}

        temp_data = {}
        global_file_id = 0
        print("도로로 병장: 파편을 썰어 특징을 다시 추출하오... (수 분 소요)")
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext == '': continue
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'rb') as f:
                        chunk_idx = 0
                        while True:
                            data = f.read(self.chunk_size)
                            if not data: break
                            if chunk_idx == 0: chunk_idx += 1; continue
                            if len(data) < self.chunk_size: data = data + b'\x00' * (self.chunk_size - len(data))
                            byte_array = np.frombuffer(data, dtype=np.uint8)
                            hist, _ = np.histogram(byte_array, bins=256, range=(0, 256))
                            ent_map = calc_entropy_map(byte_array)
                            if ext not in temp_data: temp_data[ext] = []
                            temp_data[ext].append((byte_array, hist / self.chunk_size, ent_map, global_file_id))
                            chunk_idx += 1
                        global_file_id += 1
                except: pass

        valid_exts = [ext for ext, items in temp_data.items() if len(items) >= min_class_samples]
        for i, ext in enumerate(valid_exts):
            self.ext_to_label[ext] = i; self.label_to_ext[i] = ext
        for ext in valid_exts:
            label = self.ext_to_label[ext]
            for b_arr, h_norm, e_map, f_id in temp_data[ext]:
                self.fragments.append(torch.tensor(b_arr, dtype=torch.uint8))
                self.histograms.append(torch.tensor(h_norm, dtype=torch.float32))
                self.entropy_maps.append(torch.tensor(e_map, dtype=torch.float32))
                self.labels.append(label); self.file_ids.append(f_id)

    def __len__(self): return len(self.fragments)
    def __getitem__(self, idx): return self.fragments[idx].long(), self.histograms[idx], self.entropy_maps[idx], self.labels[idx], self.file_ids[idx]

dataset = RealWorldGovDocsDataset(raw_data_dir, min_class_samples=100)
num_classes = len(dataset.ext_to_label)

# 분할 시드 고정
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==========================================
# 4. 모델 뼈대 구축 및 영혼 주입
# ==========================================
class FinalMTLModel(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=64, num_classes=3):
        super(FinalMTLModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU(); self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.hist_mlp = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64))
        self.ent_mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
        self.classifier = nn.Linear(256, num_classes)
        self.cluster_embedder = nn.Linear(256, 64)

    def forward(self, x_bytes, x_hist, x_ent):
        x = self.embedding(x_bytes).transpose(1, 2)
        x = self.pool(self.relu(self.conv1(x))).transpose(1, 2)
        x = self.transformer(x)
        global_feat = torch.cat((x.mean(dim=1), self.hist_mlp(x_hist), self.ent_mlp(x_ent)), dim=1)
        # ArcFace용 정규화 유지
        return self.classifier(global_feat), F.normalize(self.cluster_embedder(global_feat), p=2, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FinalMTLModel(num_classes=num_classes).to(device)

print(f"\n도로로 병장: 요새에서 ArcFace 지식({model_save_path})을 불러오오!")
# 이전 코드에서 딕셔너리로 저장했으므로 'model' 키값만 쏙 빼서 주입!
checkpoint = torch.load(model_save_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()
print(" -> 로딩 완료! 즉시 K-Means 조립 검증에 들어갑니다!")

# ==========================================
# 5. 데이터 식별 및 임베딩 추출
# ==========================================
all_preds, all_labels = [], []
all_top3_preds = []
test_cluster_embeds, test_file_ids = [], []

with torch.no_grad():
    for bytes_in, hist_in, ent_in, labels, file_ids in tqdm(test_loader, desc="Inference"):
        bytes_in, hist_in, ent_in = bytes_in.to(device), hist_in.to(device), ent_in.to(device)
        clf_preds, cluster_embeds = model(bytes_in, hist_in, ent_in)

        _, predicted = torch.max(clf_preds, 1)
        _, top3_preds = torch.topk(clf_preds, k=3, dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_top3_preds.extend(top3_preds.cpu().numpy())
        test_cluster_embeds.extend(cluster_embeds.cpu().numpy())
        test_file_ids.extend(file_ids.numpy())

all_labels, all_preds = np.array(all_labels), np.array(all_preds)
all_top3_preds = np.array(all_top3_preds)
class_names = [dataset.label_to_ext[i] for i in range(num_classes)]

# ==========================================
# 6. K-Means 정밀 군집화 검증!
# ==========================================
print("\n" + "="*50)
print(" 🌌 [ArcFace + K-Means] 구슬처럼 빚어진 파편들을 조립하오!")
print("="*50)

best_ext, best_nmi = None, -1
best_ext_embeds, best_ext_file_ids = [], []

for i, ext in dataset.label_to_ext.items():
    ext_indices = [idx for idx, label in enumerate(all_labels) if label == i]
    if len(ext_indices) < 10: continue

    e_embeds = np.array(test_cluster_embeds)[ext_indices]
    e_file_ids = np.array(test_file_ids)[ext_indices]

    # K-Means 실행 (정답 파일 개수를 안다는 가정하에 K값 세팅)
    num_files = len(np.unique(e_file_ids))
    if num_files <= 1: continue

    kmeans_ext = KMeans(n_clusters=num_files, random_state=42, n_init=10)
    e_labels = kmeans_ext.fit_predict(e_embeds)

    e_nmi = normalized_mutual_info_score(e_file_ids, e_labels)
    print(f" 📦 [{ext.upper()}] K-Means 조립 NMI: {e_nmi:.4f} (원본 {num_files}개)")

    if e_nmi > best_nmi:
        best_nmi = e_nmi; best_ext = ext
        best_ext_embeds, best_ext_file_ids = e_embeds, e_file_ids

print(f"\n🏆 K-Means 최고 복원 확장자: '{best_ext.upper()}' (NMI: {best_nmi:.4f})")

# ==========================================
# 7. 4대 시각화 차트 출력
# ==========================================
print("\n도로로 병장: 랩미팅 비교용 4대 지도를 전개하오!")

# [1] Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
plt.title('Top-1 Classification Map', fontsize=15, fontweight='bold'); plt.show()

# [2] Top-1 vs Top-3 Bar Chart
t1_accs, t3_accs = [], []
for i in range(num_classes):
    idx = (all_labels == i)
    t1_accs.append(np.mean(all_preds[idx] == all_labels[idx]) * 100 if np.sum(idx)>0 else 0)
    t3_accs.append(np.mean(np.any(all_top3_preds[idx] == all_labels[idx][:, None], axis=1)) * 100 if np.sum(idx)>0 else 0)

plt.figure(figsize=(16, 6))
x_axis = np.arange(num_classes)
plt.bar(x_axis-0.2, t1_accs, 0.4, label='Top-1', color='salmon')
plt.bar(x_axis+0.2, t3_accs, 0.4, label='Top-3', color='crimson')
plt.xticks(x_axis, class_names, rotation=45); plt.ylabel('Acc (%)')
plt.title('Top-1 vs Top-3 Accuracy'); plt.legend(); plt.show()

# [3] Overall t-SNE (K-Means Labels)
print("도로로 병장: 전체 은하수 지도를 그리고 있소! (잠시만 대기)")
num_files_all = len(np.unique(test_file_ids))
kmeans_all = KMeans(n_clusters=num_files_all, random_state=42, n_init=10)
all_cluster_labels = kmeans_all.fit_predict(test_cluster_embeds)
nmi_all = normalized_mutual_info_score(test_file_ids, all_cluster_labels)

indices = np.random.choice(len(test_cluster_embeds), min(2000, len(test_cluster_embeds)), replace=False)
X_emb = TSNE(n_components=2, random_state=42).fit_transform(np.array(test_cluster_embeds)[indices])
plt.figure(figsize=(10, 8))
plt.scatter(X_emb[:, 0], X_emb[:, 1], c=np.array(test_file_ids)[indices], cmap='turbo', edgecolors='k', alpha=0.6)
plt.title(f'Overall K-Means Clustering Map (NMI: {nmi_all:.4f})'); plt.colorbar(); plt.show()

# [4] Best Extension Only (Detailed K-Means)
indices_best = np.random.choice(len(best_ext_embeds), min(1000, len(best_ext_embeds)), replace=False)
X_emb_best = TSNE(n_components=2, random_state=42).fit_transform(best_ext_embeds[indices_best])
plt.figure(figsize=(10, 8))
plt.scatter(X_emb_best[:, 0], X_emb_best[:, 1], c=best_ext_file_ids[indices_best], cmap='turbo', edgecolors='k', alpha=0.8)
plt.title(f'K-Means Mastery - {best_ext.upper()} Only (NMI: {best_nmi:.4f})'); plt.show()

print("\n도로로 병장: K-Means 평가가 완료되었소! HDBSCAN과 어느 쪽이 우월한지 랩미팅에서 확인해 보시옵소서!")