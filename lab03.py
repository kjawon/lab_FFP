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
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score
from sklearn.manifold import TSNE
import torch.nn.functional as F
from collections import Counter

# ==========================================
# 1. 구글 드라이브 마운트 및 저장 경로 설정
# ==========================================
print("도로로 병장: 그림자속등불 공의 구글 드라이브를 연결 중이오!")
drive.mount('/content/drive')

save_dir = '/content/drive/MyDrive/Forensic_AI_MTL_Models'
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, 'Hybrid_MTL_ByteModel_Final.pth')

# ==========================================
# 2. 대규모 GovDocs1 데이터 폭격 (000~009.zip)
# ==========================================
print("\n도로로 병장: 훈련/테스트용 대규모 데이터를 AWS S3에서 폭격 중이오! (약 5GB)")
base_url = "https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/"
raw_data_dir = "/content/govdocs_massive"
os.makedirs(raw_data_dir, exist_ok=True)

zip_ids = [f"{i:03d}" for i in range(10)]

for zid in zip_ids:
    zip_filename = f"{zid}.zip"
    zip_path = f"/content/{zip_filename}"
    url = base_url + zip_filename

    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_data_dir)

    os.remove(zip_path) # 하드 용량 확보 인술

# ==========================================
# 3. 데이터 파편화 및 1:1:1 데이터 밸런싱
# ==========================================
class GovDocsMTLDataset(Dataset):
    def __init__(self, data_dir, chunk_size=4096):
        self.chunk_size = chunk_size
        self.fragments = []
        self.labels = []
        self.file_ids = []

        self.ext_to_label = {'.pdf': 0, '.jpg': 1, '.txt': 2}
        temp_fragments = {0: [], 1: [], 2: []}
        temp_file_ids = {0: [], 1: [], 2: []}

        print("\n도로로 병장: 파편을 썰어 라벨과 '고유 파일 ID(주민번호)'를 부여하는 중이오...")
        global_file_id_counter = 0

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.ext_to_label:
                    label = self.ext_to_label[ext]
                    filepath = os.path.join(root, file)

                    try:
                        with open(filepath, 'rb') as f:
                            chunk_idx = 0
                            while True:
                                data = f.read(self.chunk_size)
                                if not data: break
                                # [핵심] 첫 번째 조각(헤더) 제거
                                if chunk_idx == 0:
                                    chunk_idx += 1
                                    continue
                                if len(data) < self.chunk_size:
                                    data = data + b'\x00' * (self.chunk_size - len(data))

                                byte_array = np.frombuffer(data, dtype=np.uint8)
                                temp_fragments[label].append(torch.tensor(byte_array, dtype=torch.uint8))
                                temp_file_ids[label].append(global_file_id_counter)
                                chunk_idx += 1

                            global_file_id_counter += 1
                    except Exception as e: pass

        print("\n도로로 병장: 데이터 평준화(1:1:1) 작전을 개시하오!")
        min_samples = min(len(temp_fragments[0]), len(temp_fragments[1]), len(temp_fragments[2]))

        for label in [0, 1, 2]:
            all_fragments = temp_fragments[label]
            all_file_ids = temp_file_ids[label]

            np.random.seed(42)
            indices = np.random.choice(len(all_fragments), min_samples, replace=False)

            for idx in indices:
                self.fragments.append(all_fragments[idx])
                self.labels.append(label)
                self.file_ids.append(all_file_ids[idx])

        print(f"✅ 데이터 밸런싱 완수! PDF/JPG/TXT 각각 {min_samples}개씩, 총 {len(self.fragments)}개로 재구성되었소이다!")

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx):
        return self.fragments[idx].long(), self.labels[idx], self.file_ids[idx]

dataset = GovDocsMTLDataset(raw_data_dir)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==========================================
# 4. 하이브리드 MTL 쌍두사 모델 정의
# ==========================================
class HybridMTLByteModel(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=64, num_classes=3):
        super(HybridMTLByteModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Linear(128, num_classes)
        self.cluster_embedder = nn.Linear(128, 64)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)

        global_feat = x.mean(dim=1)

        clf_output = self.classifier(global_feat)
        cluster_feat = self.cluster_embedder(global_feat)
        # 군집화 벡터 L2 정규화 (코사인 거리 측정에 최적화)
        cluster_embed = F.normalize(cluster_feat, p=2, dim=1)

        return clf_output, cluster_embed

# ==========================================
# 5. 다중 작업 학습(MTL) 실행
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridMTLByteModel().to(device)

criterion_clf = nn.CrossEntropyLoss()
criterion_cluster = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
print(f"\n도로로 병장: 쌍두사 전술 훈련을 개시하오! 총 {num_epochs} 에포크!")

for epoch in range(num_epochs):
    model.train()
    total_running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True, position=0)

    for inputs, labels, file_ids in progress_bar:
        inputs, labels, file_ids = inputs.to(device), labels.to(device), file_ids.to(device)

        optimizer.zero_grad()
        clf_preds, cluster_embeds = model(inputs)

        clf_loss = criterion_clf(clf_preds, labels)

        # Triplet Loss를 위한 샘플링 인술
        batch_size_curr = inputs.size(0)
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
        progress_bar.set_postfix({'총 Loss': f"{total_loss.item():.4f}"})

    print(f" ➡️ [Epoch {epoch+1} 완료] 평균 손실: {total_running_loss/len(train_loader):.4f}")

# ==========================================
# 6. 평가, 개선된 딥즐 조립 및 명세서 출력
# ==========================================
model.eval()

all_preds, all_labels = [], []
test_cluster_embeds, test_file_ids = [], []

print("\n도로로 병장: 실전 모의고사 및 정밀 딥즐 조립을 시작하오!")
with torch.no_grad():
    for inputs, labels, file_ids in test_loader:
        inputs, labels, file_ids = inputs.to(device), labels.to(device), file_ids.to(device)
        clf_preds, cluster_embeds = model(inputs)

        _, predicted = torch.max(clf_preds, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        test_cluster_embeds.extend(cluster_embeds.cpu().numpy())
        test_file_ids.extend(file_ids.cpu().numpy())

# [핵심 수리 완료] 그물코를 대폭 조인 정밀 DBSCAN
# eps=0.05 (매우 엄격), metric='cosine' (벡터 각도 기준) 적용!
print("\n도로로 병장: 그물코를 촘촘히 조여 거대 슬라임을 해체하고 진짜 가족끼리만 정밀 조립하오!")
dbscan = DBSCAN(eps=0.05, min_samples=2, metric='cosine')
test_cluster_labels = dbscan.fit_predict(test_cluster_embeds)
nmi_score = normalized_mutual_info_score(test_file_ids, test_cluster_labels)

# ---------------------------------------------------------
# 🚨 딥즐 조립 명세서 시크릿 검열! 🚨
# ---------------------------------------------------------
print("\n=======================================================")
print(" 🕵️‍♂️ [정밀 딥즐 조립 명세서] AI가 묶어낸 파편들의 실제 출신 성분 검열!")
print("=======================================================")
unique_clusters = np.unique(test_cluster_labels)
# 노이즈(-1)를 제외한 유효한 그룹 중 5개만 샘플로 뽑아서 검열하오!
valid_clusters = [c for c in unique_clusters if c != -1][:5]

if len(valid_clusters) == 0:
    print(" ⚠️ 도로로 병장: 이런! 훈련이 더 필요하여 명확히 조립된 그룹이 아직 없소이다.")
else:
    for cluster_id in valid_clusters:
        print(f"\n📦 [AI 조립 방 번호: {cluster_id}]")
        indices = np.where(test_cluster_labels == cluster_id)[0]
        true_ids_in_cluster = np.array(test_file_ids)[indices]

        id_counts = Counter(true_ids_in_cluster)

        print(f" - 소속된 파편 개수: 총 {len(indices)}개")
        print(f" - 실제 출신 원본 파일(File ID) 번호 분포: {dict(id_counts)}")

        if len(id_counts) == 1:
            print("  🎯 도로로 병장의 극찬: 완벽하오! 단일 원본 파일 조립에 100% 성공했소이다!!")
        else:
            print("  ⚠️ 도로로 병장의 경고: 아직 미세한 불순물이 섞여 있소이다.")

# ---------------------------------------------------------
# 분류 성적표 및 t-SNE 시각화
# ---------------------------------------------------------
print("\n=======================================================")
print(" 📜 [분류 성적표] 정밀도(Precision), 재현율(Recall), F1-Score")
print("=======================================================")
class_names = ['PDF', 'JPG', 'TXT']
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

print("\n도로로 병장: 은하수(t-SNE) 지도를 출력하오!")
vis_samples = min(2000, len(test_cluster_embeds))
indices = np.random.choice(len(test_cluster_embeds), vis_samples, replace=False)

# [오타 완벽 수정] n_components=2
X_embedded = TSNE(n_components=2, random_state=42).fit_transform(np.array(test_cluster_embeds)[indices])
y_true = np.array(test_file_ids)[indices]

plt.figure(figsize=(10, 8))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_true, cmap='gist_ncar', edgecolors='k', alpha=0.7)
plt.title('File Fragment Clustering (Deepzzle Simulation) - t-SNE', fontsize=14)
plt.xlabel('t-SNE Feature 1', fontsize=12)
plt.ylabel('t-SNE Feature 2', fontsize=12)
plt.colorbar(label='File ID (Original File)')
plt.show()

# ==========================================
# 7. 최종 작전 보고서 및 구글 드라이브 저장
# ==========================================
final_clf_accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
print(f"\n✅ 최종 분류 정확도: {final_clf_accuracy:.2f}% (꼼수 없는 진짜 실력!)")
print(f"✅ 정밀 딥즐 조립 성공률(NMI Score): {nmi_score:.4f} (1에 가까울수록 완벽!)")

torch.save(model.state_dict(), model_save_path)
print(f"✅ 도로로 병장: 모든 임무 대성공! 위대한 쌍두사 모델이 드라이브에 저장되었소이다!")