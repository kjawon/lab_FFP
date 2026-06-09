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
model_save_path = os.path.join(save_dir, 'Ultimate_MTL_Deepzzle.pth')

# ==========================================
# 2. 대규모 데이터 폭격 (이번엔 5개 창고만 털어 안정성 확보! 약 2.5GB)
# ==========================================
print("\n도로로 병장: AWS S3에서 데이터를 폭격하오! (000~004.zip)")
base_url = "https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/"
raw_data_dir = "/content/govdocs_all_classes"
os.makedirs(raw_data_dir, exist_ok=True)

# 메모리 폭발 방지를 위해 5개의 zip만 사용해도 수만 개의 파편이 나옵니다!
zip_ids = [f"{i:03d}" for i in range(5)]
for zid in zip_ids:
    zip_path = f"/content/{zid}.zip"
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(base_url + f"{zid}.zip", zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_data_dir)
    os.remove(zip_path)

# ==========================================
# 3. [핵심] 모든 클래스 수용 및 히스토그램 추출 + 밸런싱
# ==========================================
class UltimateGovDocsDataset(Dataset):
    def __init__(self, data_dir, chunk_size=4096, min_class_samples=500):
        self.chunk_size = chunk_size
        self.fragments = []
        self.histograms = [] # [새로운 무기] 256차원 바이트 빈도수
        self.labels = []
        self.file_ids = []
        self.ext_to_label = {}
        self.label_to_ext = {}

        temp_data = {} # {ext: [(bytes, hist, file_id), ...]}

        print("\n도로로 병장: 세상의 모든 확장자를 탐색하며 '바이트 히스토그램'을 뽑아내고 있소이다!")
        global_file_id = 0

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext == '': continue # 확장자 없는 파일 제외

                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'rb') as f:
                        chunk_idx = 0
                        while True:
                            data = f.read(self.chunk_size)
                            if not data: break
                            if chunk_idx == 0: # 헤더 제거
                                chunk_idx += 1
                                continue
                            if len(data) < self.chunk_size:
                                data = data + b'\x00' * (self.chunk_size - len(data))

                            byte_array = np.frombuffer(data, dtype=np.uint8)

                            # [닌자 비기] 256차원 바이트 빈도수 히스토그램 생성 (압축 데이터 파훼법!)
                            hist, _ = np.histogram(byte_array, bins=256, range=(0, 256))
                            hist_norm = hist / self.chunk_size # 0~1 사이로 정규화

                            if ext not in temp_data:
                                temp_data[ext] = []
                            temp_data[ext].append((byte_array, hist_norm, global_file_id))
                            chunk_idx += 1
                        global_file_id += 1
                except: pass

        # 병력이 일정 수 이상(min_class_samples) 모인 강력한 클래스들만 참전시킴
        valid_exts = [ext for ext, items in temp_data.items() if len(items) >= min_class_samples]
        print(f"\n도로로 병장: 총 {len(temp_data)}종류의 확장자 중, 전투에 적합한 {len(valid_exts)}개의 메인 클래스를 선정했소이다!")
        print(f"참전 클래스: {valid_exts}")

        # 라벨 딕셔너리 생성
        for i, ext in enumerate(valid_exts):
            self.ext_to_label[ext] = i
            self.label_to_ext[i] = ext

        # [데이터 평준화] 가장 적은 병력에 맞춰 공평하게 1:1:1... 비율 생성
        min_samples = min([len(temp_data[ext]) for ext in valid_exts])
        print(f"도로로 병장: 꼼수 방지를 위해 모든 클래스를 {min_samples}개씩 공평하게 맞추겠소이다!")

        for ext in valid_exts:
            items = temp_data[ext]
            np.random.seed(42)
            indices = np.random.choice(len(items), min_samples, replace=False)
            label = self.ext_to_label[ext]

            for idx in indices:
                b_arr, h_norm, f_id = items[idx]
                self.fragments.append(torch.tensor(b_arr, dtype=torch.uint8))
                self.histograms.append(torch.tensor(h_norm, dtype=torch.float32))
                self.labels.append(label)
                self.file_ids.append(f_id)

        print(f"✅ 최종 정예 병력: {len(valid_exts)}개 클래스, 총 {len(self.fragments)}개 파편 준비 완료!")

    def __len__(self): return len(self.fragments)
    def __getitem__(self, idx):
        return self.fragments[idx].long(), self.histograms[idx], self.labels[idx], self.file_ids[idx]

dataset = UltimateGovDocsDataset(raw_data_dir, min_class_samples=500)
num_classes = len(dataset.ext_to_label)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==========================================
# 4. [완전 개조] 1D CNN + Transformer + Histogram 하이브리드 모델
# ==========================================
class UltimateMTLModel(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=64, num_classes=3):
        super(UltimateMTLModel, self).__init__()

        # [경로 A] 바이트 순서 분석기 (1D CNN + Transformer)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # [경로 B] 히스토그램(통계) 분석기 (압축 데이터의 맹점 파훼!)
        self.hist_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # 융합된 영혼(128 + 64 = 192차원)
        self.classifier = nn.Linear(192, num_classes)
        self.cluster_embedder = nn.Linear(192, 64)

    def forward(self, x_bytes, x_hist):
        # 경로 A 연산
        x = self.embedding(x_bytes).transpose(1, 2)
        x = self.pool(self.relu(self.conv1(x))).transpose(1, 2)
        x = self.transformer(x)
        seq_feat = x.mean(dim=1) # [Batch, 128]

        # 경로 B 연산
        hist_feat = self.hist_mlp(x_hist) # [Batch, 64]

        # 두 개의 닌자 무기 융합!
        global_feat = torch.cat((seq_feat, hist_feat), dim=1) # [Batch, 192]

        clf_output = self.classifier(global_feat)
        cluster_feat = self.cluster_embedder(global_feat)
        cluster_embed = F.normalize(cluster_feat, p=2, dim=1) # L2 정규화

        return clf_output, cluster_embed

# ==========================================
# 5. 가혹한 훈련 개시 (Margin=5.0)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UltimateMTLModel(num_classes=num_classes).to(device)

criterion_clf = nn.CrossEntropyLoss()
# [핵심] 다른 파일이면 무조건 우주 끝으로 밀어버리도록 Margin을 5.0으로 폭증시킴!
criterion_cluster = nn.TripletMarginLoss(margin=5.0, p=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
print(f"\n도로로 병장: 히스토그램과 가혹한 채찍(Margin 5.0)을 장착한 쌍두사 훈련 개시! 총 {num_epochs} 에포크!")

for epoch in range(num_epochs):
    model.train()
    total_running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True, position=0)
    for bytes_in, hist_in, labels, file_ids in progress_bar:
        bytes_in, hist_in = bytes_in.to(device), hist_in.to(device)
        labels, file_ids = labels.to(device), file_ids.to(device)

        optimizer.zero_grad()
        clf_preds, cluster_embeds = model(bytes_in, hist_in)
        clf_loss = criterion_clf(clf_preds, labels)

        # Triplet Mining
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
# 6. 실전 평가 및 K-Means 딥즐 조립
# ==========================================
model.eval()
all_preds, all_labels = [], []
test_cluster_embeds, test_file_ids = [], []

print("\n도로로 병장: 실전 모의고사 진입! 파편들을 쏟아붓고 있소이다!")
with torch.no_grad():
    for bytes_in, hist_in, labels, file_ids in test_loader:
        bytes_in, hist_in = bytes_in.to(device), hist_in.to(device)
        clf_preds, cluster_embeds = model(bytes_in, hist_in)

        _, predicted = torch.max(clf_preds, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        test_cluster_embeds.extend(cluster_embeds.cpu().numpy())
        test_file_ids.extend(file_ids.cpu().numpy())

# ==========================================
# [제1 관문] 분류(Classification) 검열
# ==========================================
print("\n" + "="*60)
print(" 📜 [제1 관문: 분류 성적표] 확장자를 얼마나 잘 맞추었는가?")
print("="*60)
class_names = [dataset.label_to_ext[i] for i in range(num_classes)]
print(classification_report(all_labels, all_preds, target_names=class_names))

final_clf_accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
print(f"✅ 최종 분류 정확도: {final_clf_accuracy:.2f}%\n")

# ==========================================
# [제2 관문] 군집화(Clustering) 검열 (K-Means 도입!)
# ==========================================
print("="*60)
print(" 🧩 [제2 관문: 딥즐 군집화 성적표] 같은 파일끼리 잘 조립했는가?")
print("="*60)
# K-Means를 사용하여 슬라임 몬스터를 원본 파일 개수만큼 강제로 찢어발깁니다!
num_true_files = len(np.unique(test_file_ids))
print(f"도로로 병장: 테스트 파편들의 실제 원본 파일 개수는 {num_true_files}개이옵니다.")
print("강력한 K-Means 인술로 정확히 방을 {0}개 만들어 파편들을 쑤셔 넣겠소이다!".format(num_true_files))

kmeans = KMeans(n_clusters=num_true_files, random_state=42, n_init=10)
test_cluster_labels = kmeans.fit_predict(test_cluster_embeds)

nmi_score = normalized_mutual_info_score(test_file_ids, test_cluster_labels)
print(f"✅ 딥즐 조립 성공률(NMI Score): {nmi_score:.4f} (1에 가까울수록 완벽!)")

print("\n🕵️‍♂️ [조립 명세서 샘플 3개 검열]")
unique_clusters = np.unique(test_cluster_labels)
for cluster_id in unique_clusters[:3]:
    indices = np.where(test_cluster_labels == cluster_id)[0]
    true_ids_in_cluster = np.array(test_file_ids)[indices]
    id_counts = Counter(true_ids_in_cluster)

    print(f"📦 [AI 조립 방 번호: {cluster_id}] | 소속 파편: {len(indices)}개 | 실제 파일 분포: {dict(id_counts)}")
    if len(id_counts) == 1:
        print("  🎯 도로로 병장: 대성공! 단일 파일 조립 완벽하오!")
    else:
        print("  ⚠️ 도로로 병장: 불순물이 약간 섞였군. 분발해야겠소.")

# ==========================================
# 은하수(t-SNE) 시각화 및 모델 저장
# ==========================================
print("\n도로로 병장: 영혼의 은하수(t-SNE) 지도를 출력하오!")
vis_samples = min(2000, len(test_cluster_embeds))
indices = np.random.choice(len(test_cluster_embeds), vis_samples, replace=False)
X_embedded = TSNE(n_components=2, random_state=42).fit_transform(np.array(test_cluster_embeds)[indices])
y_true = np.array(test_file_ids)[indices]

plt.figure(figsize=(10, 8))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_true, cmap='gist_ncar', edgecolors='k', alpha=0.7)
plt.title('File Fragment Clustering (K-Means) - t-SNE', fontsize=14)
plt.colorbar(label='Original File ID')
plt.show()

torch.save(model.state_dict(), model_save_path)
print(f"\n✅ 도로로 병장: 모든 임무가 완료되었소이다! 모델은 요새({model_save_path})에 보관되었소!")