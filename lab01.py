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

# ==========================================
# 1. 구글 드라이브 마운트 및 저장 경로 설정
# ==========================================
print("도로로 병장: 그림자속등불 공의 구글 드라이브를 연결 중이오!")
drive.mount('/content/drive')

save_dir = '/content/drive/MyDrive/Forensic_AI_Models'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'HybridByteModel_GovDocs1_10x.pth')

# ==========================================
# 2. GovDocs1 10배(000~009.zip) 연쇄 다운로드 작전!
# ==========================================
print("\n도로로 병장: 스케일 10배 확장! 총 10개의 AWS S3 데이터 창고를 연속으로 털어오겠소이다! (약 5GB 이상)")
base_url = "https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/"
data_dir = "/content/govdocs_massive"
os.makedirs(data_dir, exist_ok=True)

# 000부터 009까지 10개의 zip 파일 번호 생성
zip_ids = [f"{i:03d}" for i in range(10)]

for zid in zip_ids:
    zip_filename = f"{zid}.zip"
    zip_path = f"/content/{zip_filename}"
    url = base_url + zip_filename

    # 다운로드
    if not os.path.exists(zip_path):
        print(f" ⬇️ [{zip_filename}] 강하 중... 잠시 대기하시옵소서!")
        urllib.request.urlretrieve(url, zip_path)

    # 압축 해제
    print(f" 📦 [{zip_filename}] 파일 수천 개 전개 중!")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # 용량 관리를 위해 압축 푼 zip 파일은 즉시 폭파!
    os.remove(zip_path)
    print(f" 💥 [{zip_filename}] 용량 확보를 위해 원본 파일 폭파 완료!\n")

# ==========================================
# 3. 데이터 파편화 (헤더 제거) 및 최적화 데이터로더
# ==========================================
class GovDocsFragmentDataset(Dataset):
    def __init__(self, data_dir, chunk_size=4096):
        self.chunk_size = chunk_size
        self.fragments = []
        self.labels = []

        self.ext_to_label = {'.pdf': 0, '.jpg': 1, '.txt': 2}

        print("도로로 병장: 수만 개의 파일에서 타겟 포맷을 골라내어 4KB 파편으로 맹렬히 써는 중이오...")
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.ext_to_label:
                    filepath = os.path.join(root, file)
                    self._process_file(filepath, self.ext_to_label[ext])

        print(f" -> 🔪 작전 완료! 총 {len(self.fragments)}개의 훈련/테스트용 거대 파편 더미가 생성되었소이다!")

    def _process_file(self, filepath, label):
        try:
            with open(filepath, 'rb') as f:
                chunk_idx = 0
                while True:
                    data = f.read(self.chunk_size)
                    if not data:
                        break

                    if chunk_idx == 0:
                        chunk_idx += 1
                        continue

                    if len(data) < self.chunk_size:
                        data = data + b'\x00' * (self.chunk_size - len(data))

                    # 메모리 절약 핵심 기술 (uint8)
                    byte_array = np.frombuffer(data, dtype=np.uint8)
                    self.fragments.append(torch.tensor(byte_array, dtype=torch.uint8))
                    self.labels.append(label)
                    chunk_idx += 1
        except Exception as e:
            pass

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx):
        return self.fragments[idx].long(), self.labels[idx]

dataset = GovDocsFragmentDataset(data_dir)

# 넉넉해진 데이터! 훈련/테스트 분할 (8:2)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 데이터가 많아졌으니 배치 사이즈를 64로 늘려 속도를 더 끌어올리겠소!
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==========================================
# 4. 하이브리드 모델 정의 (1D CNN + Transformer)
# ==========================================
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

# ==========================================
# 5. 실전 학습 실행 (진행 척도 표시)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridByteModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
print(f"\n도로로 병장: 부대 차렷! 10배 커진 거대 훈련 데이터를 장착하고 총 {num_epochs} 에포크 훈련을 개시하오!")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True, position=0)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        progress_bar.set_postfix({'실시간 Loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(train_loader)
    print(f" ➡️ [Epoch {epoch+1} 완료] 평균 손실(Loss): {epoch_loss:.4f}\n")

# ==========================================
# 6. 대규모 평가 및 드라이브 저장
# ==========================================
model.eval()
correct = 0
total = 0

print("도로로 병장: 거친 훈련이 끝났소! 이제 압도적인 양의 미지 파편으로 혹독한 테스트를 진행하오...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\n✅ [10배 스케일업] 최종 파편 분류 정확도: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), save_path)
print(f"✅ 도로로 병장: 작전 대성공! 극한의 훈련을 마친 강력한 모델 가중치가 드라이브에 보관되었소이다!")
print(f"저장된 위치: {save_path}")