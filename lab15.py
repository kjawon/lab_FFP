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
print("도로로 병장: 차량 포렌식 전용 특수 부대(전체 데이터 투입형)를 편성하오!")
drive.mount('/content/drive')
save_dir = '/content/drive/MyDrive/Forensic_AI_Ultimate'
os.makedirs(save_dir, exist_ok=True)
# 이전 모델과 겹치지 않게 이름을 바꿉니다!
model_save_path = os.path.join(save_dir, 'Vehicle_TXT_Specialist_FullData.pth')

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
# 2. [전처리] 전체 데이터 로드 및 이진 분류(TXT vs Others) 라벨링
# ==========================================
def calc_entropy_map(byte_array, block_size=64):
    reshaped = byte_array.reshape(-1, block_size)
    ents = []
    for row in reshaped:
        counts = np.bincount(row, minlength=256)
        p = counts[counts > 0] / block_size
        ents.append(-np.sum(p * np.log2(p)))
    return np.array(ents, dtype=np.float32) / 8.0

class FullDataTXTSpecialistDataset(Dataset):
    def __init__(self, data_dir, chunk_size=4096, target_ext='.txt', min_class_samples=100):
        self.chunk_size = chunk_size
        self.fragments, self.histograms, self.entropy_maps = [], [], []
        self.binary_labels = [] # 1: target_ext (TXT), 0: Others
        self.arc_labels = []    # TXT 내부의 파일 ID (Others는 임시로 0 부여, 학습 땐 무시됨)
        self.original_file_ids = []

        temp_data = {}
        global_file_id = 0
        txt_file_id_counter = 0

        print(f"\n도로로 병장: 12만 개의 모든 데이터를 로드하며 '{target_ext}'에 표적 지시를 내리고 있소!")
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext == '': continue
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'rb') as f:
                        chunk_idx = 0
                        is_target = (ext == target_ext)
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
                            # 튜플에 is_target 플래그도 같이 저장!
                            temp_data[ext].append((byte_array, hist / self.chunk_size, ent_map, global_file_id, is_target))
                            chunk_idx += 1

                        if is_target:
                            txt_file_id_counter += 1
                        global_file_id += 1
                except: pass

        # 🚨 [생명줄] 알파벳 정렬(sorted) 적용하여 라벨 섞임 방지!
        valid_exts = sorted([ext for ext, items in temp_data.items() if len(items) >= min_class_samples])

        for ext in valid_exts:
            for b_arr, h_norm, e_map, f_id, is_tgt in temp_data[ext]:
                self.fragments.append(torch.tensor(b_arr, dtype=torch.uint8))
                self.histograms.append(torch.tensor(h_norm, dtype=torch.float32))
                self.entropy_maps.append(torch.tensor(e_map, dtype=torch.float32))
                self.original_file_ids.append(f_id)

                if is_tgt:
                    self.binary_labels.append(1)  # TXT는 1
                else:
                    self.binary_labels.append(0)  # 나머지는 모조리 0

        # ArcFace용 파일 ID는 오직 TXT 파일에 대해서만 발급합니다.
        # TXT의 원래 global_file_id를 0부터 시작하는 순차적 ID로 변환
        txt_orig_fids = sorted(list(set([f_id for f_id, is_tgt in zip(self.original_file_ids, self.binary_labels) if is_tgt == 1])))
        self.num_txt_files = len(txt_orig_fids)
        self.fid_to_arc = {orig: arc_id for arc_id, orig in enumerate(txt_orig_fids)}

        for f_id, is_tgt in zip(self.original_file_ids, self.binary_labels):
            if is_tgt == 1:
                self.arc_labels.append(self.fid_to_arc[f_id])
            else:
                self.arc_labels.append(0) # 0으로 둡니다. (학습 땐 txt_mask로 걸러내서 쓰이지 않음)

        total = len(self.binary_labels)
        txt_count = sum(self.binary_labels)
        print(f"✅ 장전 완료! 총 {total}개 파편 (TXT: {txt_count}개, 쓰레기: {total - txt_count}개)")
        print(f"   원본 TXT 파일은 {self.num_txt_files}개이옵니다!")

    def __len__(self): return len(self.fragments)
    def __getitem__(self, idx):
        return (self.fragments[idx].long(), self.histograms[idx], self.entropy_maps[idx],
                self.binary_labels[idx], self.arc_labels[idx], self.original_file_ids[idx])

dataset = FullDataTXTSpecialistDataset(raw_data_dir, target_ext='.txt')

# 훈련: 80%
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 검증: 100% 전체 파편 투입
full_loader = DataLoader(dataset, batch_size=128, shuffle=False)

# ==========================================
# 3. 모델 및 ArcFace 정의 (2 클래스 이진 분류)
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
    # num_classes = 2 (0: Others, 1: TXT)
    def __init__(self, vocab_size=256, embed_dim=64, num_classes=2):
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
model = FinalMTLModel(num_classes=2).to(device)
arcface_layer = ArcMarginProduct(in_features=64, out_features=dataset.num_txt_files).to(device)
criterion_ce = nn.CrossEntropyLoss()

# ==========================================
# 4. 특수 부대 맞춤형 투트랙 훈련 (15/15 Epochs)
# ==========================================
print("\n" + "="*50)
print("🔥 [STAGE 1] 전체 데이터 대상 TXT 색출 특훈 (15 Epochs)")
print("="*50)
optimizer_stage1 = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(15):
    model.train()
    pbar = tqdm(train_loader, desc=f"S1 Ep {epoch+1}/15")
    for bytes_in, hist_in, ent_in, bin_labels, _, _ in pbar:
        bytes_in, hist_in, ent_in, bin_labels = bytes_in.to(device), hist_in.to(device), ent_in.to(device), bin_labels.to(device)
        optimizer_stage1.zero_grad()
        clf_preds, _ = model(bytes_in, hist_in, ent_in)
        loss = criterion_ce(clf_preds, bin_labels)
        loss.backward(); optimizer_stage1.step()

print("\n" + "="*50)
print("❄️ [STAGE 2] 뼈대 동결! 오직 TXT끼리의 정밀 조립 특훈 (15 Epochs)")
print("="*50)
for param in model.parameters(): param.requires_grad = False
for param in model.cluster_embedder.parameters(): param.requires_grad = True

optimizer_stage2 = optim.Adam(list(model.cluster_embedder.parameters()) + list(arcface_layer.parameters()), lr=0.001)
for epoch in range(15):
    model.train(); arcface_layer.train()
    pbar = tqdm(train_loader, desc=f"S2 Ep {epoch+1}/15")
    for bytes_in, hist_in, ent_in, bin_labels, arc_labels, _ in pbar:
        # [핵심] 배치 안에 섞여있는 쓰레기는 무시하고, TXT(bin_labels == 1) 파편만 골라내어 조립 훈련!
        txt_mask = (bin_labels == 1)
        if txt_mask.sum() == 0: continue

        bytes_txt = bytes_in[txt_mask].to(device)
        hist_txt = hist_in[txt_mask].to(device)
        ent_txt = ent_in[txt_mask].to(device)
        arc_labels_txt = arc_labels[txt_mask].to(device)

        optimizer_stage2.zero_grad()
        _, cluster_embeds = model(bytes_txt, hist_txt, ent_txt)
        loss = criterion_ce(arcface_layer(cluster_embeds, arc_labels_txt), arc_labels_txt)
        loss.backward(); optimizer_stage2.step()

torch.save({'model': model.state_dict(), 'arcface': arcface_layer.state_dict()}, model_save_path)
print("\n✅ 전체 데이터 기반 차량 텍스트 전용 모델이 완성되었소이다!")

# ==========================================
# 5. 전수 조사 및 실전 파이프라인 검증 (100% 데이터)
# ==========================================
model.eval()
all_preds, all_bin_labels = [], []
all_embeds, all_original_fids = [], []

print("\n도로로 병장: 100% 데이터를 투입하여 TXT를 색출하고 조립을 개시하오!")
with torch.no_grad():
    for bytes_in, hist_in, ent_in, bin_labels, _, orig_fids in tqdm(full_loader, desc="Inference (100%)"):
        bytes_in, hist_in, ent_in = bytes_in.to(device), hist_in.to(device), ent_in.to(device)

        clf_preds, cluster_embeds = model(bytes_in, hist_in, ent_in)
        _, predicted = torch.max(clf_preds, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_bin_labels.extend(bin_labels.numpy())
        all_embeds.extend(cluster_embeds.cpu().numpy())
        all_original_fids.extend(orig_fids.numpy())

all_bin_labels, all_preds = np.array(all_bin_labels), np.array(all_preds)
all_embeds, all_original_fids = np.array(all_embeds), np.array(all_original_fids)

# --------------------------------------------------
# 🗺️ [1차 관문] 이진 분류(TXT 필터링) 성적
# --------------------------------------------------
print("\n" + "="*60)
print(" 🛡️ [1차 분류] 쓰레기 더미 속 TXT 식별망 방어력 검증")
print("="*60)
print(classification_report(all_bin_labels, all_preds, target_names=["Others (Trash)", "TXT (Target)"]))

# --------------------------------------------------
# 🧩 [2차 관문] 추출된 TXT 데이터 정밀 조립 (HDBSCAN)
# --------------------------------------------------
print("\n" + "="*60)
print(" 🧩 [2차 조립] 1차 방어망을 통과한 파편들을 묶어내오!")
print("="*60)

# AI가 'TXT(1)'라고 예측하여 1차 관문을 통과한 파편들만 선별
predicted_txt_indices = np.where(all_preds == 1)[0]
txt_embeds = all_embeds[predicted_txt_indices]
txt_fids = all_original_fids[predicted_txt_indices]

num_files = len(np.unique(txt_fids))
print(f" -> 1차 관문 통과 파편: {len(txt_embeds)}개 (실제 원본 파일: {num_files}개 추정)")

# HDBSCAN 조립 개시
hdbscan_txt = HDBSCAN(min_cluster_size=5, min_samples=3)
txt_cluster_labels = hdbscan_txt.fit_predict(txt_embeds)
txt_nmi = normalized_mutual_info_score(txt_fids, txt_cluster_labels)

print(f" 🏆 [TXT 전용] HDBSCAN 조립 최종 성공률(NMI): {txt_nmi:.4f}")

# --------------------------------------------------
# 📊 [시각화] 차량 포렌식 특화 랩미팅 차트
# --------------------------------------------------
print("\n도로로 병장: 선택과 집중이 낳은 완벽한 결과를 시각화하오!")

# [시각화 1] 이진 분류 혼동 행렬
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(all_bin_labels, all_preds), annot=True, fmt='d', cmap='Reds', xticklabels=["Others", "TXT"], yticklabels=["Others", "TXT"])
plt.title('Stage 1: TXT Binary Classification Matrix (Full Data)', fontsize=14, fontweight='bold')
plt.xlabel('AI Prediction'); plt.ylabel('Actual Format'); plt.show()

# [시각화 2] 챔피언(TXT) 은하수 지도
# 10,000개까지 늘려서 빽빽한 우주를 보여줍니다!
plot_limit = min(10000, len(txt_embeds))
indices_plot = np.random.choice(len(txt_embeds), plot_limit, replace=False)
X_emb_txt = TSNE(n_components=2, random_state=42).fit_transform(txt_embeds[indices_plot])

plt.figure(figsize=(12, 10))
plt.scatter(X_emb_txt[:, 0], X_emb_txt[:, 1], c=txt_fids[indices_plot], cmap='turbo', edgecolors='k', alpha=0.9, s=30)
plt.title(f'Vehicle Forensics Mastery - TXT Only (NMI: {txt_nmi:.4f}, Points: {plot_limit})', fontsize=16, fontweight='bold')
plt.colorbar(label='Original TXT File ID')
plt.show()

print("\n✅ 도로로 병장: 12만 개의 파편 더미에서 TXT만 완벽하게 걸러내어 조립하는 파이프라인이 완성되었소이다!")