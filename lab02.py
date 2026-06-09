import os
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. 구글 드라이브 마운트 및 모델 부활
# ==========================================
print("도로로 병장: 그림자속등불 공의 구글 드라이브를 다시 연결하오!")
drive.mount('/content/drive')
save_path = '/content/drive/MyDrive/Forensic_AI_Models/HybridByteModel_GovDocs1_10x.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 구조 재정의 (저장할 때와 완전히 똑같은 뼈대가 있어야 하옵니다)
class HybridByteModel(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=64, num_classes=3):
        super(HybridByteModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

model = HybridByteModel().to(device)

# [핵심] 드라이브에 저장된 가중치를 불러와 모델에 장착!
print("도로로 병장: 훈련된 AI의 영혼(가중치)을 다시 주입하고 있소이다...")
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval() # 평가 모드로 전환
print(" -> 모델 부활 성공!")

# ==========================================
# 2. 테스트용 데이터(000.zip) 신속 확보 및 파편화
# ==========================================
print("\n도로로 병장: 평가를 위한 테스트 데이터를 AWS S3에서 신속히 조달하겠소이다!")
url = "https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/000.zip"
zip_path = "/content/000.zip"
data_dir = "/content/govdocs_test"

if not os.path.exists(zip_path):
    urllib.request.urlretrieve(url, zip_path)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

class GovDocsTestDataset(Dataset):
    def __init__(self, data_dir, chunk_size=4096):
        self.chunk_size = chunk_size
        self.fragments = []
        self.labels = []
        self.ext_to_label = {'.pdf': 0, '.jpg': 1, '.txt': 2}

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.ext_to_label:
                    filepath = os.path.join(root, file)
                    self._process_file(filepath, self.ext_to_label[ext])

    def _process_file(self, filepath, label):
        try:
            with open(filepath, 'rb') as f:
                chunk_idx = 0
                while True:
                    data = f.read(self.chunk_size)
                    if not data: break
                    if chunk_idx == 0:
                        chunk_idx += 1
                        continue
                    if len(data) < self.chunk_size:
                        data = data + b'\x00' * (self.chunk_size - len(data))

                    byte_array = np.frombuffer(data, dtype=np.uint8)
                    self.fragments.append(torch.tensor(byte_array, dtype=torch.uint8))
                    self.labels.append(label)
                    chunk_idx += 1
        except:
            pass

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx):
        return self.fragments[idx].long(), self.labels[idx]

print("도로로 병장: 테스트용 파편을 썰고 있소이다...")
test_dataset = GovDocsTestDataset(data_dir)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==========================================
# 3. 실전 모의고사 채점 및 성적표 출력
# ==========================================
print("\n도로로 병장: AI가 적군을 어떻게 오인했는지 정밀 분석을 시작하오!")
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

class_names = ['PDF', 'JPG', 'TXT']

print("\n=======================================================")
print(" 📜 [작전 결과 보고서] 정밀도(Precision), 재현율(Recall), F1-Score")
print("=======================================================")
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

print("\n도로로 병장: 오분류 지도(Confusion Matrix)를 출력하오!")

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.title('File Fragment Classification - Confusion Matrix', fontsize=14)
plt.ylabel('True Format (Real Label)', fontsize=12)
plt.xlabel('Predicted Format (AI Prediction)', fontsize=12)
plt.show()

print("도로로 병장: 모든 임무가 무사히 재개되었소이다!")