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
from sklearn.cluster import HDBSCAN
from sklearn.metrics import normalized_mutual_info_score
from sklearn.manifold import TSNE
import torch.nn.functional as F

# ==========================================
# 1. 구글 드라이브 마운트 및 요새 연결
# ==========================================
print("도로로 병장: 요새를 연결하고 전면 재학습을 준비하오!")
drive.mount('/content/drive')
save_dir = '/content/drive/MyDrive/Forensic_AI_Ultimate'
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, 'ArcFace_TwoStage_Deepzzle_30Epochs.pth')

# ==========================================
# 2. 데이터 조달
# ==========================================
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
# 3. 데이터 전처리 (sorted 적용된 철통 방어 버전!)
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
        print("\n도로로 병장: 새로운 정답표(알파벳 순서)를 기준으로 파편들을 썰고 있소이다!")
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

        # 🚨 [생명줄] sorted로 확장자 정답 번호 영구 고정!
        valid_exts = sorted([ext for ext, items in temp_data.items() if len(items) >= min_class_samples])
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

        # ArcFace용 파일 ID 번호도 영구 고정!
        unique_file_ids = sorted(list(set(self.file_ids)))
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
class_names = [dataset.label_to_ext[i] for i in range(num_classes)]

# 훈련을 위해 80:20 분할 (훈련은 80%로 진행)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# 검증용으로는 100% 전체 파편 투입!
full_loader = DataLoader(dataset, batch_size=128, shuffle=False)

# ==========================================
# 4. 모델 뼈대 및 ArcFace 계층 정의
# ==========================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.s = s; self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-6)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s

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
arcface_layer = ArcMarginProduct(in_features=64, out_features=dataset.num_arc_classes).to(device)
criterion_ce = nn.CrossEntropyLoss()

# ==========================================
# 5. [핵심] 30 에포크 지옥 훈련 (Two-Stage)
# ==========================================
stage1_epochs = 30
stage2_epochs = 30

print("\n" + "="*50)
print(f"🔥 [STAGE 1] 분류기 전용 특훈 개시 ({stage1_epochs} Epochs)")
print("="*50)
optimizer_stage1 = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(stage1_epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"S1 Ep {epoch+1}/{stage1_epochs}")
    for bytes_in, hist_in, ent_in, labels, _, _ in pbar:
        bytes_in, hist_in, ent_in, labels = bytes_in.to(device), hist_in.to(device), ent_in.to(device), labels.to(device)
        optimizer_stage1.zero_grad()
        clf_preds, _ = model(bytes_in, hist_in, ent_in)
        loss = criterion_ce(clf_preds, labels)
        loss.backward(); optimizer_stage1.step()

print("\n" + "="*50)
print(f"❄️ [STAGE 2] 뼈대 동결! ArcFace 군집화 특훈 개시 ({stage2_epochs} Epochs)")
print("="*50)
for param in model.parameters(): param.requires_grad = False
for param in model.cluster_embedder.parameters(): param.requires_grad = True

optimizer_stage2 = optim.Adam(list(model.cluster_embedder.parameters()) + list(arcface_layer.parameters()), lr=0.001)
for epoch in range(stage2_epochs):
    model.train(); arcface_layer.train()
    pbar = tqdm(train_loader, desc=f"S2 Ep {epoch+1}/{stage2_epochs}")
    for bytes_in, hist_in, ent_in, _, _, arc_labels in pbar:
        bytes_in, hist_in, ent_in, arc_labels = bytes_in.to(device), hist_in.to(device), ent_in.to(device), arc_labels.to(device)
        optimizer_stage2.zero_grad()
        _, cluster_embeds = model(bytes_in, hist_in, ent_in)
        loss = criterion_ce(arcface_layer(cluster_embeds, arc_labels), arc_labels)
        loss.backward(); optimizer_stage2.step()

# 새로운 30에포크 두뇌 저장!
torch.save({'model': model.state_dict(), 'arcface': arcface_layer.state_dict()}, model_save_path)
print("\n✅ 60에포크 지옥 훈련을 마친 신버전의 뇌가 무사히 저장되었소이다!")

# ==========================================
# 6. 전수 조사 및 실전 파이프라인 검증 (100% 데이터)
# ==========================================
model.eval()
all_preds, all_labels, all_top3_preds, all_embeds, all_file_ids = [], [], [], [], []

print("\n도로로 병장: 100% 전체 파편에 대한 검증을 즉각 실시하오!")
with torch.no_grad():
    for bytes_in, hist_in, ent_in, labels, file_ids, _ in tqdm(full_loader, desc="Inference (100%)"):
        bytes_in, hist_in, ent_in = bytes_in.to(device), hist_in.to(device), ent_in.to(device)

        clf_preds, cluster_embeds = model(bytes_in, hist_in, ent_in)
        _, predicted = torch.max(clf_preds, 1)
        _, top3_preds = torch.topk(clf_preds, k=3, dim=1)

        all_preds.extend(predicted.cpu().numpy()); all_labels.extend(labels.numpy())
        all_top3_preds.extend(top3_preds.cpu().numpy())
        all_embeds.extend(cluster_embeds.cpu().numpy()); all_file_ids.extend(file_ids.numpy())

all_labels, all_preds, all_top3_preds = np.array(all_labels), np.array(all_preds), np.array(all_top3_preds)
all_embeds, all_file_ids = np.array(all_embeds), np.array(all_file_ids)

# --------------------------------------------------
# 🗺️ [결과 확인] 1차 분류 성적
# --------------------------------------------------
print("\n" + "="*60)
print(" 🗺️ [1차 분류 성적] 60 에포크 특훈의 결과를 확인하시오!")
print("="*60)
print(classification_report(all_labels, all_preds, target_names=class_names))

# ==========================================
# 7. 계층적 딥즐 조립 파이프라인 전개
# ==========================================
valid_indices = [i for i in range(len(all_labels)) if all_labels[i] in all_top3_preds[i]]
filtered_labels = all_labels[valid_indices]
filtered_embeds = all_embeds[valid_indices]
filtered_file_ids = all_file_ids[valid_indices]

best_ext, best_nmi = None, -1
best_ext_embeds, best_ext_file_ids = [], []
nmi_results = {}

print("\n" + "="*60)
print(" 🧩 [2차 조립] 100% 우량 파편을 각 확장자 방에서 파일 ID별로 묶어내오!")
print("="*60)

for i, ext in dataset.label_to_ext.items():
    ext_idx = np.where(filtered_labels == i)[0]
    if len(ext_idx) < 10: continue

    room_embeds = filtered_embeds[ext_idx]
    room_file_ids = filtered_file_ids[ext_idx]
    num_files_in_room = len(np.unique(room_file_ids))
    if num_files_in_room <= 1: continue

    hdbscan_room = HDBSCAN(min_cluster_size=5, min_samples=3)
    room_cluster_labels = hdbscan_room.fit_predict(room_embeds)

    room_nmi = normalized_mutual_info_score(room_file_ids, room_cluster_labels)
    nmi_results[ext] = room_nmi
    print(f" 📦 [{ext.upper()}] ➡️ NMI: {room_nmi:.4f} (원본 {num_files_in_room}개, 파편 {len(room_embeds)}개 조립 완료!)")

    if room_nmi > best_nmi:
        best_nmi = room_nmi; best_ext = ext
        best_ext_embeds, best_ext_file_ids = room_embeds, room_file_ids

# --------------------------------------------------
# 📊 [시각화] 모든 랩미팅 차트 출력
# --------------------------------------------------
t1_accs, t3_accs = [], []
for i in range(num_classes):
    idx = (all_labels == i)
    t1_accs.append(np.mean(all_preds[idx] == all_labels[idx]) * 100 if np.sum(idx)>0 else 0)
    t3_accs.append(np.mean(np.any(all_top3_preds[idx] == all_labels[idx][:, None], axis=1)) * 100 if np.sum(idx)>0 else 0)

x = np.arange(num_classes)
plt.figure(figsize=(16, 6))
plt.bar(x - 0.2, t1_accs, 0.4, label='Top-1 Accuracy', color='salmon')
plt.bar(x + 0.2, t3_accs, 0.4, label='Top-3 Accuracy', color='crimson')
plt.title('Stage 1: Top-1 vs Top-3 Classification Accuracy (30 Epochs)', fontsize=16, fontweight='bold')
plt.xticks(x, class_names, rotation=45); plt.ylabel('Acc (%)'); plt.legend(); plt.show()

plt.figure(figsize=(14, 12))
sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
plt.title('Stage 1: Confusion Matrix', fontsize=16, fontweight='bold'); plt.show()

extensions = list(nmi_results.keys())
nmi_scores = [nmi_results[e]*100 for e in extensions]
plt.figure(figsize=(14, 6))
sns.barplot(x=extensions, y=nmi_scores, palette="viridis")
plt.title("Stage 2: Assembly Success Rate (NMI %) by File ID", fontsize=16, fontweight='bold')
plt.ylim(0, 110); plt.xticks(rotation=45)
for i, v in enumerate(nmi_scores): plt.text(i, v + 1, f"{v:.1f}", ha='center')
plt.show()

plot_limit = min(10000, len(best_ext_embeds))
indices_best = np.random.choice(len(best_ext_embeds), plot_limit, replace=False)
X_emb_best = TSNE(n_components=2, random_state=42).fit_transform(best_ext_embeds[indices_best])
plt.figure(figsize=(12, 10))
plt.scatter(X_emb_best[:, 0], X_emb_best[:, 1], c=best_ext_file_ids[indices_best], cmap='turbo', edgecolors='k', alpha=0.9, s=30)
plt.title(f'Pipeline Mastery - {best_ext.upper()} (NMI: {best_nmi:.4f}, Points: {plot_limit})', fontsize=16, fontweight='bold')
plt.colorbar(label='Original File ID'); plt.show()

print("\n✅ 도로로 병장: 60 에포크 대수술이 끝났소이다! 완벽히 부활한 랩미팅용 성적표를 확인하시옵소서!")