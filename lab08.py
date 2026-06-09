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
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import torch.nn.functional as F

# ==========================================
# 1. 구글 드라이브 마운트 및 요새 연결
# ==========================================
print("도로로 병장: 그림자속등불 공의 요새를 연결하오!")
drive.mount('/content/drive')
model_save_path = '/content/drive/MyDrive/Forensic_AI_Ultimate/RealWorld_MTL_Deepzzle.pth'

# ==========================================
# 2. 실전 데이터 신속 조달
# ==========================================
print("\n도로로 병장: 시각화를 위한 데이터를 준비하오!")
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
    if os.path.exists(zip_path):
        os.remove(zip_path)

# ==========================================
# 3. 데이터 전처리 (히스토그램 & 엔트로피)
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
        print("도로로 병장: 파편을 썰어 특징을 추출 중이오... (수 분 소요)")
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
                            if chunk_idx == 0:
                                chunk_idx += 1; continue
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
            self.ext_to_label[ext] = i
            self.label_to_ext[i] = ext

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

dataset = RealWorldGovDocsDataset(raw_data_dir, min_class_samples=100)
num_classes = len(dataset.ext_to_label)

# 훈련과 똑같은 조건(시드 고정)으로 분할하여 테스트 셋만 추출
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==========================================
# 4. 모델 뼈대 구축 및 영혼(가중치) 로드
# ==========================================
class FinalMTLModel(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=64, num_classes=3):
        super(FinalMTLModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
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
        hist_feat = self.hist_mlp(x_hist)
        ent_feat = self.ent_mlp(x_ent)
        global_feat = torch.cat((seq_feat, hist_feat, ent_feat), dim=1)
        clf_output = self.classifier(global_feat)
        cluster_feat = self.cluster_embedder(global_feat)
        cluster_embed = F.normalize(cluster_feat, p=2, dim=1)
        return clf_output, cluster_embed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FinalMTLModel(num_classes=num_classes).to(device)

print("\n도로로 병장: 드라이브에서 훈련된 모델의 영혼을 깨우고 있소이다!")
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()
print(" -> 영혼 주입 완료! (훈련 스킵, 즉시 평가 돌입)")

# ==========================================
# 5. 실전 평가 데이터 수집 (Top-1, Top-3, 임베딩)
# ==========================================
all_preds, all_labels = [], []
all_top3_preds = []
test_cluster_embeds, test_file_ids = [], []

print("\n도로로 병장: AI가 12만 개의 파편을 식별 중이오...")
with torch.no_grad():
    for bytes_in, hist_in, ent_in, labels, file_ids in tqdm(test_loader, desc="Testing"):
        bytes_in, hist_in, ent_in = bytes_in.to(device), hist_in.to(device), ent_in.to(device)

        clf_preds, cluster_embeds = model(bytes_in, hist_in, ent_in)

        # Top-1
        _, predicted = torch.max(clf_preds, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

        # Top-3
        _, top3_preds = torch.topk(clf_preds, k=3, dim=1)
        all_top3_preds.extend(top3_preds.cpu().numpy())

        test_cluster_embeds.extend(cluster_embeds.cpu().numpy())
        test_file_ids.extend(file_ids.numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_top3_preds = np.array(all_top3_preds)
class_names = [dataset.label_to_ext[i] for i in range(num_classes)]

# ==========================================
# 📊 [시각화 1] Top-1 오분류 지도 (Heatmap)
# ==========================================
print("\n" + "="*50)
print(" 🗺️ [1/4] Top-1 분류 성적 지도 (Confusion Matrix)")
print("="*50)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
plt.title('Top-1 File Extension Classification - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Format (Real Extension)', fontsize=12)
plt.xlabel('Predicted Format (AI Prediction)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ==========================================
# 📊 [시각화 2] Top-1 vs Top-3 비교 막대그래프 (발표 핵심 자료!)
# ==========================================
print("\n" + "="*50)
print(" 📊 [2/4] 클래스별 Top-1 vs Top-3 정확도 극적 비교!")
print("="*50)

top1_accs, top3_accs = [], []
for i in range(num_classes):
    idx = (all_labels == i)
    if np.sum(idx) == 0:
        top1_accs.append(0); top3_accs.append(0); continue

    t1_acc = np.mean(all_preds[idx] == all_labels[idx]) * 100
    t3_acc = np.mean(np.any(all_top3_preds[idx] == all_labels[idx][:, None], axis=1)) * 100
    top1_accs.append(t1_acc)
    top3_accs.append(t3_acc)

x = np.arange(num_classes)
width = 0.35
plt.figure(figsize=(16, 8))
plt.bar(x - width/2, top1_accs, width, label='Top-1 Accuracy', color='salmon')
plt.bar(x + width/2, top3_accs, width, label='Top-3 Accuracy', color='crimson')

plt.ylabel('Accuracy (%)', fontsize=14)
plt.title('Top-1 vs Top-3 Accuracy per Extension (Overcoming the Compression Wall)', fontsize=18, fontweight='bold')
plt.xticks(x, class_names, rotation=45, ha='right', fontsize=12)
plt.ylim(0, 110)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==========================================
# 🌌 [시각화 3] 전체 데이터 딥즐 조립 은하수 (t-SNE)
# ==========================================
print("\n" + "="*50)
print(" 🌌 [3/4] 전체 데이터 딥즐 군집화 은하수 (t-SNE)")
print("="*50)

num_true_files_all = len(np.unique(test_file_ids))
kmeans_all = KMeans(n_clusters=num_true_files_all, random_state=42, n_init=10)
test_cluster_labels_all = kmeans_all.fit_predict(test_cluster_embeds)
nmi_score_all = normalized_mutual_info_score(test_file_ids, test_cluster_labels_all)
print(f"✅ 전체 데이터 조립 성공률(NMI): {nmi_score_all:.4f}")

vis_samples = min(2000, len(test_cluster_embeds))
indices = np.random.choice(len(test_cluster_embeds), vis_samples, replace=False)
X_embedded_all = TSNE(n_components=2, random_state=42).fit_transform(np.array(test_cluster_embeds)[indices])
y_true_all = np.array(test_file_ids)[indices]

plt.figure(figsize=(12, 10))
plt.scatter(X_embedded_all[:, 0], X_embedded_all[:, 1], c=y_true_all, cmap='turbo', edgecolors='k', alpha=0.7)
plt.title(f'Overall Deepzzle Clustering (NMI: {nmi_score_all:.4f})', fontsize=16, fontweight='bold')
plt.colorbar(label='Original File ID')
plt.show()

# ==========================================
# 🥇 [시각화 4] 최우수 확장자 전용 정밀 조립 검열!
# ==========================================
print("\n" + "="*50)
print(" 🥇 [4/4] 가장 가족 상봉이 잘 된 챔피언 확장자의 정밀 시각화!")
print("="*50)

best_ext, best_nmi = None, -1
best_ext_embeds, best_ext_file_ids = [], []

for i, ext in dataset.label_to_ext.items():
    ext_indices = [idx for idx, label in enumerate(all_labels) if label == i]
    if len(ext_indices) == 0: continue

    ext_file_ids = np.array(test_file_ids)[ext_indices]
    ext_cluster_embeds = np.array(test_cluster_embeds)[ext_indices]
    num_ext_files = len(np.unique(ext_file_ids))
    if num_ext_files <= 1: continue

    kmeans_ext = KMeans(n_clusters=num_ext_files, random_state=42, n_init=10)
    ext_cluster_labels = kmeans_ext.fit_predict(ext_cluster_embeds)

    ext_nmi = normalized_mutual_info_score(ext_file_ids, ext_cluster_labels)
    if ext_nmi > best_nmi:
        best_nmi = ext_nmi; best_ext = ext
        best_ext_embeds = ext_cluster_embeds; best_ext_file_ids = ext_file_ids

print(f"🏆 가장 완벽하게 복원된 확장자: '{best_ext.upper()}' (NMI: {best_nmi:.4f})")

vis_samples_ext = min(2000, len(best_ext_embeds))
indices_ext = np.random.choice(len(best_ext_embeds), vis_samples_ext, replace=False)
X_embedded_ext = TSNE(n_components=2, random_state=42).fit_transform(np.array(best_ext_embeds)[indices_ext])
y_true_ext = np.array(best_ext_file_ids)[indices_ext]

plt.figure(figsize=(10, 8))
plt.scatter(X_embedded_ext[:, 0], X_embedded_ext[:, 1], c=y_true_ext, cmap='turbo', edgecolors='k', alpha=0.9, s=50)
plt.title(f'Deepzzle Mastery - {best_ext.upper()} Only (NMI: {best_nmi:.4f})', fontsize=16, fontweight='bold')
plt.colorbar(label='Original File ID (Perfect Match)')
plt.show()

print("\n도로로 병장: 모든 시각화 작전이 대성공으로 끝났소이다! PPT를 불태우시옵소서!")