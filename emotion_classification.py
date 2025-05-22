import torch
import torch.backends
import torch.backends.mps
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer

# 모델 정의 (위에서 작성한 CNN 모델)
class CNN(nn.Module):
  def __init__(self, vocab_size, embedding_dim, n_kernels, kernel_sizes, output_dim, dropout, pad_idx):
    super().__init__()
    self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=embedding_dim,
                                  padding_idx=pad_idx)
    
    self.convs = nn.ModuleList([
      nn.Conv2d(in_channels=1,
                out_channels=n_kernels,
                kernel_size=(ksize, embedding_dim))
      for ksize in kernel_sizes
    ])
    
    # 추가된 레이어
    self.fc1 = nn.Linear(len(kernel_sizes) * n_kernels, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, output_dim)
    
    self.dropout = nn.Dropout(dropout)
    self.batch_norm1 = nn.BatchNorm1d(512)
    self.batch_norm2 = nn.BatchNorm1d(256)

  def forward(self, review):
    review = review.to(device)
    embedded = self.embedding(review.to(device))
    embedded = embedded.unsqueeze(1)
    
    conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
    pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
    
    cat = torch.cat(pooled, dim=1)
    
    # 추가된 레이어 통과
    x = self.dropout(cat)
    x = self.fc1(x)
    x = self.batch_norm1(x)
    x = F.relu(x)
    x = self.dropout(x)
    
    x = self.fc2(x)
    x = self.batch_norm2(x)
    x = F.relu(x)
    x = self.dropout(x)
    
    x = self.fc3(x)
    return x

vocab = torch.load('vocab.pth')

# 모델 로드
model = CNN(vocab_size=4999, embedding_dim=512, n_kernels=256, kernel_sizes=[3, 4, 5], output_dim=7, dropout=0.5, pad_idx=vocab['<PAD>'])
state_dict = torch.load('best-model.pt', map_location=torch.device("mps"))
state_dict['embedding.weight'] = state_dict['embedding.weight'][:5000, :]
print(f"best-model.weight shape: {state_dict['embedding.weight'].shape}")

model.load_state_dict(state_dict)  # 모델 가중치 파일
model.eval()

# device 설정
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

# tokenizer 설정 (토큰화 및 텍스트 전처리)
tokenizer = get_tokenizer('basic_english')

def emotion_classifying(input_diary: str):
  tokens = tokenizer(input_diary)
  token_ids = [vocab[token] if token in vocab else vocab['<UNK>'] for token in tokens]  # UNK 처리
    
  # 텐서로 변환 후, GPU로 이동
  text_tensor = torch.LongTensor(token_ids).to(device)
    
  # 배치 차원 추가 (batch_size=1)
  text_tensor = text_tensor.unsqueeze(0)
    
  # 모델 예측
  with torch.no_grad():
    output = model(text_tensor)
    prediction = torch.argmax(output, dim=1).item()  # 가장 높은 확률의 클래스를 예측
    
  return prediction