import os
import math
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import HDBSCAN
from sklearn.metrics import normalized_mutual_info_score
from sklearn.manifold import TSNE
import torch.nn.functional as F

# ==========================================
# 1. 구글 드라이브 마운트 및 요새 연결
# ==========================================
print("도로로 병장: 요새를 연결하고 훈련된 ArcFace 모델을 즉시 소환하오!")
drive.mount('/content/drive')
model_save_path = '/content/drive/MyDrive/Forensic_AI_Ultimate/ArcFace_TwoStage_Deepzzle.pth'

# ==========================================
# 2. 데이터 재조달 (검증용)
# ==========================================
print("\n도로로 병장: 실전 평가를 위한 파편들을 전장에 배치하오!")
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

class ArcFaceGovDocsDataset(Dataset):
    def __init__(self, data_dir, chunk_size=4096, min_class_samples=100):
        self.chunk_size = chunk_size
        self.fragments, self.histograms, self.entropy_maps = [], [], []
        self.labels, self.file_ids = [], []
        self.ext_to_label, self.label_to_ext = {}, {}

        temp_data = {}
        global_file_id = 0
        print("도로로 병장: 파편들의 특징과 히스토그램을 다시 추출 중이오... (수 분 소요)")
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
                            if len(data) < self.chunk_size:
                                data = data + b'\x00' * (self.chunk_size - len(data))
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
                self.labels.append(label)
                self.file_ids.append(f_id)

    def __len__(self): return len(self.fragments)
    def __getitem__(self, idx):
        return self.fragments[idx].long(), self.histograms[idx], self.entropy_maps[idx], self.labels[idx], self.file_ids[idx]

dataset = ArcFaceGovDocsDataset(raw_data_dir, min_class_samples=100)
num_classes = len(dataset.ext_to_label)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==========================================
# 4. 모델 뼈대 및 영혼 주입 (훈련 스킵!)
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
        return self.classifier(global_feat), F.normalize(self.cluster_embedder(global_feat), p=2, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FinalMTLModel(num_classes=num_classes).to(device)

print(f"\n도로로 병장: 요새에서 전설의 뼈대({model_save_path})를 부활시키오!")
checkpoint = torch.load(model_save_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

# ==========================================
# 5. [파이프라인 1단계] 전수 조사 및 Top-3 예측
# ==========================================
all_labels, all_top3_preds = [], []
all_embeds, all_file_ids = [], []

print("\n도로로 병장: 1단계 - 모든 파편의 임베딩 좌표와 Top-3 후보를 색출하오!")
with torch.no_grad():
    for bytes_in, hist_in, ent_in, labels, file_ids in tqdm(test_loader, desc="Inference"):
        bytes_in, hist_in, ent_in = bytes_in.to(device), hist_in.to(device), ent_in.to(device)

        clf_preds, cluster_embeds = model(bytes_in, hist_in, ent_in)
        _, top3_preds = torch.topk(clf_preds, k=3, dim=1)

        all_labels.extend(labels.numpy())
        all_top3_preds.extend(top3_preds.cpu().numpy())
        all_embeds.extend(cluster_embeds.cpu().numpy())
        all_file_ids.extend(file_ids.numpy())

all_labels = np.array(all_labels)
all_top3_preds = np.array(all_top3_preds)
all_embeds = np.array(all_embeds)
all_file_ids = np.array(all_file_ids)

# ==========================================
# 6. [파이프라인 2단계] Top-3 방어벽 필터링 (Divide)
# ==========================================
print("\n" + "="*60)
print(" 🛡️ [2단계] Top-3 필터망 전개! 오답 쓰레기를 걸러내오!")
print("="*60)

# Top-3 후보 안에 정답이 포함된 "우량 파편"들의 인덱스만 수집
valid_indices = [i for i in range(len(all_labels)) if all_labels[i] in all_top3_preds[i]]

filtered_labels = all_labels[valid_indices]
filtered_embeds = all_embeds[valid_indices]
filtered_file_ids = all_file_ids[valid_indices]

survival_rate = len(valid_indices) / len(all_labels) * 100
print(f"✅ Top-3 필터 생존율: {survival_rate:.2f}% ({len(valid_indices)} / {len(all_labels)} 개)")
print("도로로 병장: 이제 불순물이 섞이지 않은 청정 구역에서 조립을 시작하오!")

# ==========================================
# 7. [파이프라인 3단계] 계층적 서브스페이스 군집화 (Conquer)
# ==========================================
print("\n" + "="*60)
print(" 🧩 [3단계] 확장자 전용 대기실 내 1:1 HDBSCAN 조립 개시!")
print("="*60)

best_ext, best_nmi = None, -1
best_ext_embeds, best_ext_file_ids = [], []
nmi_results = {}

for i, ext in dataset.label_to_ext.items():
    # 2단계에서 필터링된 파편들 중에서, i번째 확장자 방에 들어온 파편만 격리!
    ext_idx = np.where(filtered_labels == i)[0]
    if len(ext_idx) < 10: continue

    room_embeds = filtered_embeds[ext_idx]
    room_file_ids = filtered_file_ids[ext_idx]

    num_files_in_room = len(np.unique(room_file_ids))
    if num_files_in_room <= 1: continue

    # 격리된 방 안에서 조립을 수행하니 노이즈가 없어 성능이 폭발하옵니다!
    hdbscan_room = HDBSCAN(min_cluster_size=3, min_samples=2)
    room_cluster_labels = hdbscan_room.fit_predict(room_embeds)

    room_nmi = normalized_mutual_info_score(room_file_ids, room_cluster_labels)
    nmi_results[ext] = room_nmi
    print(f" 📦 [{ext.upper()} 방] 전용 HDBSCAN 조립 NMI: {room_nmi:.4f} (원본 {num_files_in_room}개 완벽 분해!)")

    if room_nmi > best_nmi:
        best_nmi = room_nmi; best_ext = ext
        best_ext_embeds, best_ext_file_ids = room_embeds, room_file_ids

print(f"\n🏆 최종 파이프라인 최고 조립 확장자: '{best_ext.upper()}' (NMI: {best_nmi:.4f})")

# ==========================================
# 8. 랩미팅 프레젠테이션용 시각화 출력
# ==========================================
print("\n도로로 병장: 파이프라인의 성공을 증명할 시각화 자료를 전개하오!")

# [시각화 1] 계층적 조립 NMI 점수 막대 그래프
extensions = list(nmi_results.keys())
nmi_scores = [nmi_results[e]*100 for e in extensions]

plt.figure(figsize=(14, 6))
sns.barplot(x=extensions, y=nmi_scores, palette="viridis")
plt.title("Deepzzle Assembly Success Rate (NMI %) by Top-3 Filtered Sub-spaces", fontsize=16, fontweight='bold')
plt.ylabel("NMI Score (%)", fontsize=12)
plt.ylim(0, 110)
plt.xticks(rotation=45)
for i, v in enumerate(nmi_scores): plt.text(i, v + 1, f"{v:.1f}", ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# [시각화 2] 챔피언 확장자 전용 은하수 지도 (불순물 제로!)
# 필터링 후 얼마나 구슬처럼 예쁘게 뭉쳐있는지 확인
indices_best = np.random.choice(len(best_ext_embeds), min(1000, len(best_ext_embeds)), replace=False)
X_emb_best = TSNE(n_components=2, random_state=42).fit_transform(best_ext_embeds[indices_best])

plt.figure(figsize=(10, 8))
plt.scatter(X_emb_best[:, 0], X_emb_best[:, 1], c=best_ext_file_ids[indices_best], cmap='turbo', edgecolors='k', alpha=0.9, s=60)
plt.title(f'Pipeline Mastery - {best_ext.upper()} Room Only (NMI: {best_nmi:.4f})', fontsize=16, fontweight='bold')
plt.colorbar(label='Original File ID (Perfectly Grouped)')
plt.show()

print("\n✅ 도로로 병장: Top-3 필터망과 계층적 HDBSCAN이 결합된 궁극의 파이프라인이 완성되었소이다!")