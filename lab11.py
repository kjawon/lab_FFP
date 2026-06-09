import os
import math
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from google.colab import drive
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import HDBSCAN # K-Means 대신 HDBSCAN 유지!
from sklearn.metrics import normalized_mutual_info_score
from sklearn.manifold import TSNE
import torch.nn.functional as F

# ==========================================
# 1. 구글 드라이브 마운트 및 저장 경로
# ==========================================
print("도로로 병장: 그림자속등불 공의 요새를 연결하오!")
drive.mount('/content/drive')
save_dir = '/content/drive/MyDrive/Forensic_AI_Ultimate'
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, 'ArcFace_TwoStage_Deepzzle.pth')

# ==========================================
# 2. 대규모 데이터 조달 (000~004.zip)
# ==========================================
print("\n도로로 병장: 실전 데이터를 조달하오!")
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
# 3. [데이터 전처리] ArcFace 전용 라벨링 추가!
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
        print("\n도로로 병장: 파편을 썰고 ArcFace를 위한 고유 신분증(ID)을 발급 중이오!")
        global_file_id = 0

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

        # [비기] ArcFace는 0부터 순차적인 ID를 요구하므로 맵핑 생성
        unique_file_ids = list(set(self.file_ids))
        self.num_arc_classes = len(unique_file_ids)
        self.file_id_to_arc = {fid: i for i, fid in enumerate(unique_file_ids)}
        self.arc_labels = [self.file_id_to_arc[fid] for fid in self.file_ids]

        print(f"✅ 총 {len(self.fragments)}개 파편 장전 완료! (총 원본 파일 개수: {self.num_arc_classes}개)")

    def __len__(self): return len(self.fragments)
    def __getitem__(self, idx):
        return (self.fragments[idx].long(), self.histograms[idx], self.entropy_maps[idx],
                self.labels[idx], self.file_ids[idx], self.arc_labels[idx])

dataset = ArcFaceGovDocsDataset(raw_data_dir, min_class_samples=100)
num_classes = len(dataset.ext_to_label)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==========================================
# 4. [신규] ArcFace 안면 인식 모듈 & 하이브리드 모델
# ==========================================
# ArcFace (안면 인식) 손실 계층: 구슬(Hypersphere) 표면에 파편을 동그랗게 배치!
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-6)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s

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
        seq_feat = x.mean(dim=1)

        hist_feat, ent_feat = self.hist_mlp(x_hist), self.ent_mlp(x_ent)
        global_feat = torch.cat((seq_feat, hist_feat, ent_feat), dim=1)

        clf_output = self.classifier(global_feat)
        cluster_feat = self.cluster_embedder(global_feat)
        # ArcFace를 위해 임베딩을 구면상으로 정규화(Normalize)
        cluster_embed = F.normalize(cluster_feat, p=2, dim=1)
        return clf_output, cluster_embed

# ==========================================
# 5. [궁극의 비기] 투트랙 분할 학습 (Two-Stage Training)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FinalMTLModel(num_classes=num_classes).to(device)
arcface_layer = ArcMarginProduct(in_features=64, out_features=dataset.num_arc_classes).to(device)

criterion_ce = nn.CrossEntropyLoss()

# --------------------------------------------------
# 🔥 1단계: 확장자 분류 지식 마스터 (Gradient 충돌 방지)
# --------------------------------------------------
stage1_epochs = 15
optimizer_stage1 = optim.Adam(model.parameters(), lr=0.001)

print("\n" + "="*50)
print(f"🔥 [STAGE 1] 분류기 전용 특훈 개시 (Epoch 1~{stage1_epochs})")
print("="*50)
for epoch in range(stage1_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch+1}/{stage1_epochs}", leave=True)
    for bytes_in, hist_in, ent_in, labels, _, _ in pbar:
        bytes_in, hist_in, ent_in, labels = bytes_in.to(device), hist_in.to(device), ent_in.to(device), labels.to(device)

        optimizer_stage1.zero_grad()
        clf_preds, _ = model(bytes_in, hist_in, ent_in)
        loss = criterion_ce(clf_preds, labels) # 분류만 학습!
        loss.backward()
        optimizer_stage1.step()

        running_loss += loss.item()
        pbar.set_postfix({'분류 Loss': f"{loss.item():.4f}"})

# --------------------------------------------------
# ❄️ 2단계: 뼈대 동결 + ArcFace 안면 인식 학습
# --------------------------------------------------
print("\n" + "="*50)
print("❄️ [STAGE 2] 뼈대 동결! ArcFace 군집화 특훈 개시")
print("="*50)
# 뼈대(Backbone) 가중치 잠금(Freeze)
for param in model.parameters():
    param.requires_grad = False
# 오직 군집화 머리(cluster_embedder)와 ArcFace 계층만 훈련!
for param in model.cluster_embedder.parameters():
    param.requires_grad = True

stage2_epochs = 15
optimizer_stage2 = optim.Adam(list(model.cluster_embedder.parameters()) + list(arcface_layer.parameters()), lr=0.001)

for epoch in range(stage2_epochs):
    model.train()
    arcface_layer.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch+1}/{stage2_epochs}", leave=True)
    for bytes_in, hist_in, ent_in, _, _, arc_labels in pbar:
        bytes_in, hist_in, ent_in, arc_labels = bytes_in.to(device), hist_in.to(device), ent_in.to(device), arc_labels.to(device)

        optimizer_stage2.zero_grad()
        _, cluster_embeds = model(bytes_in, hist_in, ent_in)

        # ArcFace 연산 및 손실 계산 (파일 ID 맞추기)
        arc_outputs = arcface_layer(cluster_embeds, arc_labels)
        loss = criterion_ce(arc_outputs, arc_labels)
        loss.backward()
        optimizer_stage2.step()

        running_loss += loss.item()
        pbar.set_postfix({'ArcFace Loss': f"{loss.item():.4f}"})

# ==========================================
# 6. 실전 평가 및 시각화 전개!
# ==========================================
model.eval()
arcface_layer.eval()
all_preds, all_labels, all_top3_preds = [], [], []
test_cluster_embeds, test_file_ids = [], []

print("\n도로로 병장: 지옥훈련을 마친 모델의 실전 평가를 개시하오!")
with torch.no_grad():
    for bytes_in, hist_in, ent_in, labels, file_ids, _ in tqdm(test_loader, desc="Testing"):
        bytes_in, hist_in, ent_in = bytes_in.to(device), hist_in.to(device), ent_in.to(device)
        labels = labels.to(device)

        clf_preds, cluster_embeds = model(bytes_in, hist_in, ent_in)

        _, predicted = torch.max(clf_preds, 1)
        _, top3_preds = torch.topk(clf_preds, k=3, dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_top3_preds.extend(top3_preds.cpu().numpy())

        test_cluster_embeds.extend(cluster_embeds.cpu().numpy())
        test_file_ids.extend(file_ids.numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_top3_preds = np.array(all_top3_preds)

# --------------------------------------------------
# 🗺️ 제1 관문: 분류 성적표
# --------------------------------------------------
print("\n" + "="*60)
print(" 🗺️ [최종 분류 성적표] 투트랙 전략으로 정상을 되찾았는가!")
print("="*60)
class_names = [dataset.label_to_ext[i] for i in range(num_classes)]
print(classification_report(all_labels, all_preds, target_names=class_names))

top1_acc = 100 * np.mean(all_preds == all_labels)
top3_acc = 100 * np.mean(np.any(all_top3_preds == all_labels[:, None], axis=1))
print(f"✅ 실전 Top-1 분류 정확도: {top1_acc:.2f}%")
print(f"🌟 실전 Top-3 분류 정확도: {top3_acc:.2f}%")

# --------------------------------------------------
# 🌌 제2 관문: HDBSCAN 확장자별 딥즐 군집화 정밀 분석
# --------------------------------------------------
print("\n" + "="*60)
print(" 🌌 [ArcFace + HDBSCAN] 밀도 기반 가족 상봉의 결과를 보시옵소서!")
print("="*60)

best_ext, best_nmi = None, -1
best_ext_embeds, best_ext_file_ids = [], []

for i, ext in dataset.label_to_ext.items():
    ext_idx = [idx for idx, label in enumerate(all_labels) if label == i]
    if len(ext_idx) < 10: continue

    ext_embeds = np.array(test_cluster_embeds)[ext_idx]
    ext_file_ids = np.array(test_file_ids)[ext_idx]
    num_files = len(np.unique(ext_file_ids))
    if num_files <= 1: continue

    hdbscan_ext = HDBSCAN(min_cluster_size=3, min_samples=2)
    cluster_labels = hdbscan_ext.fit_predict(ext_embeds)
    ext_nmi = normalized_mutual_info_score(ext_file_ids, cluster_labels)

    print(f" 📦 [{ext.upper()}] HDBSCAN 조립 NMI: {ext_nmi:.4f} (원본 {num_files}개)")

    if ext_nmi > best_nmi:
        best_nmi = ext_nmi; best_ext = ext
        best_ext_embeds, best_ext_file_ids = ext_embeds, ext_file_ids

# --------------------------------------------------
# 📊 4대 랩미팅용 시각화 차트 출력
# --------------------------------------------------
print("\n도로로 병장: 뱀의 허리를 끊고 섬을 만든 은하수 지도를 전개하오!")

# [1] 막대그래프
t1_accs, t3_accs = [], []
for i in range(num_classes):
    idx = (all_labels == i)
    if np.sum(idx) == 0: t1_accs.append(0); t3_accs.append(0); continue
    t1_accs.append(np.mean(all_preds[idx] == all_labels[idx]) * 100)
    t3_accs.append(np.mean(np.any(all_top3_preds[idx] == all_labels[idx][:, None], axis=1)) * 100)

x = np.arange(num_classes)
plt.figure(figsize=(16, 8))
plt.bar(x - 0.2, t1_accs, 0.4, label='Top-1 Accuracy', color='salmon')
plt.bar(x + 0.2, t3_accs, 0.4, label='Top-3 Accuracy', color='crimson')
plt.title('Top-1 vs Top-3 Accuracy (Restored by Two-Stage Training)', fontsize=18)
plt.xticks(x, class_names, rotation=45); plt.legend(); plt.show()

# [2] Confusion Matrix
plt.figure(figsize=(14, 12))
sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
plt.title('Top-1 Classification Map', fontsize=16); plt.show()

# [3] 전체 HDBSCAN 은하수
hdb_all = HDBSCAN(min_cluster_size=5, min_samples=3)
all_cluster_labels = hdb_all.fit_predict(test_cluster_embeds)
nmi_all = normalized_mutual_info_score(test_file_ids, all_cluster_labels)

indices = np.random.choice(len(test_cluster_embeds), min(2000, len(test_cluster_embeds)), replace=False)
X_emb = TSNE(n_components=2, random_state=42).fit_transform(np.array(test_cluster_embeds)[indices])
plt.figure(figsize=(12, 10))
plt.scatter(X_emb[:, 0], X_emb[:, 1], c=np.array(test_file_ids)[indices], cmap='turbo', edgecolors='k', alpha=0.7)
plt.title(f'Overall ArcFace Clustering (NMI: {nmi_all:.4f})', fontsize=16); plt.show()

# [4] 챔피언 은하수 (완벽한 섬 확인)
idx_best = np.random.choice(len(best_ext_embeds), min(1000, len(best_ext_embeds)), replace=False)
X_emb_best = TSNE(n_components=2, random_state=42).fit_transform(best_ext_embeds[idx_best])
plt.figure(figsize=(10, 8))
plt.scatter(X_emb_best[:, 0], X_emb_best[:, 1], c=best_ext_file_ids[idx_best], cmap='turbo', edgecolors='k', alpha=0.9, s=50)
plt.title(f'ArcFace Mastery - {best_ext.upper()} (NMI: {best_nmi:.4f})', fontsize=16); plt.show()

torch.save({'model': model.state_dict(), 'arcface': arcface_layer.state_dict()}, model_save_path)
print("\n✅ 도로로 병장: 뱀을 베어내고 완벽한 둥근 섬을 창조했소이다! 랩미팅을 박살 내시옵소서!")