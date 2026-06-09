import os
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
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.manifold import TSNE
import torch.nn.functional as F
from collections import Counter

# ==========================================
# 1. 구글 드라이브 마운트 및 저장 경로
# ==========================================
print("도로로 병장: 그림자속등불 공의 요새(드라이브)를 연결하오!")
drive.mount('/content/drive')
save_dir = '/content/drive/MyDrive/Forensic_AI_Ultimate'
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, 'Final_Top3_Deepzzle.pth')

# ==========================================
# 2. 대규모 데이터 조달 (000~004.zip)
# ==========================================
print("\n도로로 병장: AWS S3에서 데이터를 조달하오! (000~004.zip)")
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
# 3. [데이터 전처리] 히스토그램 + 엔트로피 추출 및 데이터 최대화
# ==========================================
# 4096바이트를 64바이트 단위로 쪼개어 복잡도(엔트로피)를 계산하는 함수
def calc_entropy_map(byte_array, block_size=64):
    reshaped = byte_array.reshape(-1, block_size)
    ents = []
    for row in reshaped:
        counts = np.bincount(row, minlength=256)
        p = counts[counts > 0] / block_size
        ents.append(-np.sum(p * np.log2(p)))
    return np.array(ents, dtype=np.float32) / 8.0

class FinalGovDocsDataset(Dataset):
    def __init__(self, data_dir, chunk_size=4096, min_class_samples=500, max_per_class=12000):
        self.chunk_size = chunk_size
        self.fragments = []
        self.histograms = []
        self.entropy_maps = []
        self.labels = []
        self.file_ids = []
        self.ext_to_label = {}
        self.label_to_ext = {}
        self.class_weights = None

        temp_data = {}

        print("\n도로로 병장: 수만 개의 파편에서 3대 특징(순서, 빈도, 엔트로피)을 추출 중이오! (수 분 소요)")
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
                            if chunk_idx == 0:  # 헤더(첫 4KB) 무조건 제거
                                chunk_idx += 1
                                continue
                            if len(data) < self.chunk_size:
                                data = data + b'\x00' * (self.chunk_size - len(data))

                            byte_array = np.frombuffer(data, dtype=np.uint8)
                            hist, _ = np.histogram(byte_array, bins=256, range=(0, 256))
                            hist_norm = hist / self.chunk_size
                            ent_map = calc_entropy_map(byte_array)

                            if ext not in temp_data:
                                temp_data[ext] = []
                            temp_data[ext].append((byte_array, hist_norm, ent_map, global_file_id))
                            chunk_idx += 1
                        global_file_id += 1
                except: pass

        # 병력이 일정 이상인 클래스만 참전
        valid_exts = [ext for ext, items in temp_data.items() if len(items) >= min_class_samples]
        print(f"\n도로로 병장: 전투에 적합한 {len(valid_exts)}개의 메인 클래스 선정 완료!")

        for i, ext in enumerate(valid_exts):
            self.ext_to_label[ext] = i
            self.label_to_ext[i] = ext

        class_counts = []
        print(f"도로로 병장: 각 클래스별 최대 {max_per_class}개 한도 내에서 모든 병력을 긁어모으겠소!")

        for ext in valid_exts:
            items = temp_data[ext]
            np.random.seed(42)

            take_count = min(len(items), max_per_class)
            indices = np.random.choice(len(items), take_count, replace=False)
            class_counts.append(take_count)
            label = self.ext_to_label[ext]

            for idx in indices:
                b_arr, h_norm, e_map, f_id = items[idx]
                self.fragments.append(torch.tensor(b_arr, dtype=torch.uint8))
                self.histograms.append(torch.tensor(h_norm, dtype=torch.float32))
                self.entropy_maps.append(torch.tensor(e_map, dtype=torch.float32))
                self.labels.append(label)
                self.file_ids.append(f_id)

        # [소수 민족 보호 인술] 클래스 불균형 해소를 위한 가중치 계산
        class_counts = np.array(class_counts)
        total_samples = np.sum(class_counts)
        weights = total_samples / (len(valid_exts) * class_counts)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

        print(f"✅ 최종 정예 병력: 총 {len(self.fragments)}개 파편 및 가중치 장전 완료!")

    def __len__(self): return len(self.fragments)
    def __getitem__(self, idx):
        return self.fragments[idx].long(), self.histograms[idx], self.entropy_maps[idx], self.labels[idx], self.file_ids[idx]

dataset = FinalGovDocsDataset(raw_data_dir, min_class_samples=500, max_per_class=12000)
num_classes = len(dataset.ext_to_label)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==========================================
# 4. [궁극의 3-Way 개조] 하이브리드 MTL 모델
# ==========================================
class FinalMTLModel(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=64, num_classes=3):
        super(FinalMTLModel, self).__init__()
        # 1. 시퀀스 분석 (CNN + Transformer)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 2. 통계 분석 (Histogram)
        self.hist_mlp = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64))
        # 3. 압축/암호화 분석 (Entropy Map)
        self.ent_mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))

        # 3가지 특징 융합 (128 + 64 + 64 = 256)
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

# ==========================================
# 5. 가중치 폭격 훈련 개시 (Epoch 30)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FinalMTLModel(num_classes=num_classes).to(device)

class_weights = dataset.class_weights.to(device)
criterion_clf = nn.CrossEntropyLoss(weight=class_weights) # 소수 클래스 페널티 강화
criterion_cluster = nn.TripletMarginLoss(margin=5.0, p=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30
print(f"\n도로로 병장: 가중치 채찍을 손에 쥐고 {num_epochs} 에포크의 대규모 훈련을 개시하오!")

for epoch in range(num_epochs):
    model.train()
    total_running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True, position=0)
    for bytes_in, hist_in, ent_in, labels, file_ids in progress_bar:
        bytes_in, hist_in, ent_in = bytes_in.to(device), hist_in.to(device), ent_in.to(device)
        labels, file_ids = labels.to(device), file_ids.to(device)

        optimizer.zero_grad()
        clf_preds, cluster_embeds = model(bytes_in, hist_in, ent_in)
        clf_loss = criterion_clf(clf_preds, labels)

        # 군집화를 위한 Triplet 추출
        batch_size_curr = bytes_in.size(0)
        triplets = []
        for i in range(batch_size_curr):
            anchor_idx = i
            anchor_file_id = file_ids[i].item()
            pos_indices = (file_ids == anchor_file_id).nonzero().flatten()
            pos_indices = pos_indices[pos_indices != anchor_idx]
            neg_indices = (file_ids != anchor_file_id).nonzero().flatten()

            if len(pos_indices) > 0 and len(neg_indices) > 0:
                pos_idx = np.random.choice(pos_indices.cpu().numpy())
                neg_idx = np.random.choice(neg_indices.cpu().numpy())
                triplets.append((cluster_embeds[anchor_idx], cluster_embeds[pos_idx], cluster_embeds[neg_idx]))

        if len(triplets) == 0:
            cluster_loss = torch.tensor(0.0).to(device)
        else:
            anchors, positives, negatives = zip(*triplets)
            anchors, positives, negatives = torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
            cluster_loss = criterion_cluster(anchors, positives, negatives)

        total_loss = clf_loss + cluster_loss
        total_loss.backward()
        optimizer.step()

        total_running_loss += total_loss.item()
        progress_bar.set_postfix({'분류Loss': f"{clf_loss.item():.4f}", '군집Loss': f"{cluster_loss.item():.4f}"})

    print(f" ➡️ [Epoch {epoch+1} 완료] 평균 손실: {total_running_loss/len(train_loader):.4f}")

# ==========================================
# 6. 실전 평가 (Top-1 & Top-3 동시 계산!)
# ==========================================
model.eval()
all_preds, all_labels = [], []
test_cluster_embeds, test_file_ids = [], []

top1_correct = 0
top3_correct = 0
total_samples = 0

print("\n도로로 병장: Top-3 관대함 인술을 전개하여 최종 평가를 진행하오!")
with torch.no_grad():
    for bytes_in, hist_in, ent_in, labels, file_ids in test_loader:
        bytes_in, hist_in, ent_in = bytes_in.to(device), hist_in.to(device), ent_in.to(device)
        labels, file_ids = labels.to(device), file_ids.to(device)

        clf_preds, cluster_embeds = model(bytes_in, hist_in, ent_in)

        # [기존] Top-1 예측
        _, predicted = torch.max(clf_preds, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        top1_correct += (predicted == labels).sum().item()

        # 🎯 [신규] Top-3 예측 계산 (압축 파일의 모호함 타파!)
        _, top3_preds = torch.topk(clf_preds, k=3, dim=1)
        labels_reshaped = labels.view(-1, 1)
        correct_in_top3 = (top3_preds == labels_reshaped).any(dim=1)

        top3_correct += correct_in_top3.sum().item()
        total_samples += labels.size(0)

        test_cluster_embeds.extend(cluster_embeds.cpu().numpy())
        test_file_ids.extend(file_ids.cpu().numpy())

# ==========================================
# 🗺️ [제1 관문] 분류 성적 시각화 지도
# ==========================================
final_top1_accuracy = 100 * (top1_correct / total_samples)
final_top3_accuracy = 100 * (top3_correct / total_samples)

print("\n" + "="*60)
print(" 🗺️ [최종 분류 성적표] 압축 데이터의 맹점을 뚫었는가?!")
print("="*60)
class_names = [dataset.label_to_ext[i] for i in range(num_classes)]
print(classification_report(all_labels, all_preds, target_names=class_names))

print(f"✅ 엄격한 Top-1 분류 정확도: {final_top1_accuracy:.2f}%")
print(f"🌟 실전용 Top-3 분류 정확도: {final_top3_accuracy:.2f}% (놀라운 상승!)")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'File Classification Map (Top-1: {final_top1_accuracy:.1f}%, Top-3: {final_top3_accuracy:.1f}%)', fontsize=16)
plt.ylabel('True Format (Real Extension)', fontsize=12)
plt.xlabel('Predicted Format (AI Prediction)', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# ==========================================
# 🌌 [제2 관문] 딥즐 군집화 성적 및 은하수 시각화
# ==========================================
print("\n" + "="*60)
print(" 🌌 [딥즐 군집화 성적] 파일별 가족 상봉 작전 검열!")
print("="*60)

num_true_files = len(np.unique(test_file_ids))
kmeans = KMeans(n_clusters=num_true_files, random_state=42, n_init=10)
test_cluster_labels = kmeans.fit_predict(test_cluster_embeds)

nmi_score = normalized_mutual_info_score(test_file_ids, test_cluster_labels)
print(f"✅ 딥즐 조립 성공률(NMI Score): {nmi_score:.4f} (1에 가까울수록 완벽!)")

vis_samples = min(2000, len(test_cluster_embeds))
indices = np.random.choice(len(test_cluster_embeds), vis_samples, replace=False)
X_embedded = TSNE(n_components=2, random_state=42).fit_transform(np.array(test_cluster_embeds)[indices])
y_true = np.array(test_file_ids)[indices]

plt.figure(figsize=(12, 10))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_true, cmap='turbo', edgecolors='k', alpha=0.8)
plt.title(f'Deepzzle Clustering Map (NMI: {nmi_score:.4f})', fontsize=16)
plt.colorbar(label='Original File ID')
plt.show()

torch.save(model.state_dict(), model_save_path)
print(f"\n✅ 도로로 병장: 모든 임무가 완료되었소이다! 요새({model_save_path})에 모델을 봉인하오!")