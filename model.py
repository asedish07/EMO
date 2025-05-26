<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from torchtext import datasets
from torchtext.vocab import build_vocab_from_iterator
import random
import numpy as np
import pandas as pd
from collections import Counter
from typing import List
from kiwipiepy import Kiwi
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용할 디바이스: {device}')

KERNEL_SIZE = [3, 4, 5]
MAX_VOCAB_SIZE = 25000
SEED = 42
BATCH_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

kiwi = Kiwi()

def most_frequent_number(lst):
    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0][0]
    
df = pd.read_excel('data/dataset.xlsx')
token_count = []
for i in range(len(df)):
  token_count.append(len(kiwi.tokenize(df['Sentence'][i], normalize_coda=False, split_complex=False, blocklist=None)))
print(f"데이터 1문장 평균 형태소 갯수: {most_frequent_number(token_count)}")


def postprocess_morphemes(morphemes):
    # "안녕하세요" -> "안녕하세요" (하 +세요 -> 안녕하세요)
    morphemes = merge_greetings(morphemes)
    
    return morphemes

def merge_greetings(morphemes):
    """인사말 '안녕 하세요'와 같은 부분을 합침"""
    merged = []
    i = 0
    while i < len(morphemes):
        # '안녕' + '하' + '세요' -> '안녕하세요'
        if morphemes[i] == '안녕' and i + 2 < len(morphemes) and morphemes[i + 1] == '하' and morphemes[i + 2] == '세요':
            merged.append('안녕하세요')
            i += 3
        else:
            merged.append(morphemes[i])
            i += 1
    return merged

def tokenizer(text):
  # 형태소로 나누기
  token_tmp = kiwi.tokenize(text, normalize_coda=False, split_complex=False, blocklist=None)
  token = []
  for t in token_tmp:
    token.append(t.form)
  token = postprocess_morphemes(token)
  
  # 패딩 토큰 추가
  if len(token) < max(KERNEL_SIZE):
    for i in range(0, max(KERNEL_SIZE)-len(token)):
      token.append('<PAD>') # 커널 사이즈보다 문장의 길이가 작은 경우 에러 방지
  return token

emo = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
emotion_to_label = {
  '공포': 0,
  '놀람': 1,
  '분노': 2,
  '슬픔': 3,
  '중립': 4,
  '행복': 5,
  '혐오': 6
}
emo_count = {0: 5468, 1: 5898, 2: 5665, 3: 5267, 4: 4830, 5: 6037, 6: 5429}

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

print(f"train_data: {len(train_df)}")
print(f"test_data: {len(test_df)}")
print(f"test_data_head: {test_df.head()}")


# class Vocab:
#   def __init__(self, tokens_list, min_freq=1):
#     counter = Counter()
#     for tokens in tokens_list:
#       counter.update(tokens)

#       # Special tokens 먼저 추가
#       self.itos = ['<PAD>', '<UNK>']
#       self.stoi = {'<PAD>': 0, '<UNK>': 1}

#       for token, freq in counter.items():
#         if freq >= min_freq and token not in self.stoi:
#           self.stoi[token] = len(self.itos)
#           self.itos.append(token)

#   def encode(self, tokens):
#     return [self.stoi.get(tok, self.stoi['<UNK>']) for tok in tokens]

#   def decode(self, indices):
#     return [self.itos[i] for i in indices]

# class TextDataset(Dataset):
#   def __init__(self, df, vocab, max_len=50):
#     self.texts = df['Sentence'].apply(tokenizer).tolist()  # 텍스트 토큰화
#     self.labels = df['Emotion'].tolist()  # 레이블
#     self.vocab = vocab  # Vocab 객체
#     self.max_len = max_len  # 문장의 최대 길이

#   def __len__(self):
#     return len(self.texts)

#   def __getitem__(self, idx):
#     token_ids = self.vocab.encode(self.texts[idx])  # 토큰 ID로 변환
#     tensor = torch.tensor(token_ids[:self.max_len])  # 최대 길이로 자르고 텐서로 변환
#     label = torch.tensor(self.labels[idx], dtype=torch.float)  # 레이블 텐서
#     return tensor, label

# def collate_fn(batch):
#   texts, labels = zip(*batch)
#   texts = pad_sequence(texts, batch_first=True, padding_value=0)
#   labels = torch.stack(labels)
#   return texts, labels

# tokenized_sentences = df['Sentence'].apply(tokenizer).tolist()
# vocab = Vocab(tokenized_sentences)

# lengths = [len(tokens) for tokens in tokenized_sentences]
# print(f"데이터 1문장 평균 토큰 갯수: {sum(lengths) / len(lengths)}")

# train_dataset = TextDataset(train_data, vocab)
# test_dataset = TextDataset(test_data, vocab)

# 텍스트와 레이블을 나누어 훈련 데이터셋과 테스트 데이터셋으로 사용
train_data = list(zip(train_df['Sentence'], train_df['Emotion']))
test_data = list(zip(test_df['Sentence'], test_df['Emotion']))

# 단어 집합 구축을 위한 토큰화 함수
def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)  # tokenizer를 사용하여 토큰화
        
# train_data로 단어 집합 생성 (unk_token 명시적으로 설정)
vocab = build_vocab_from_iterator(
    yield_tokens(train_data),
    max_tokens=MAX_VOCAB_SIZE,
    specials=["<PAD>", "<UNK>"],  # <unk>를 special token으로 설정
)

# <UNK>를 기본적으로 처리할 토큰으로 설정
vocab.set_default_index(vocab["<UNK>"])

# 텍스트를 인덱스로 변환하는 파이프라인
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]  # <UNK> 처리

# 레이블을 숫자로 변환하는 파이프라인 (7개의 클래스)
def label_pipeline(label):
    return emotion_to_label[label]  # 감정 이름을 숫자로 변환

# 데이터셋을 파이프라인 처리하여 텍스트와 레이블을 처리
train_data = [(text_pipeline(text), label_pipeline(label)) for text, label in train_data]
test_data = [(text_pipeline(text), label_pipeline(label)) for text, label in test_data]

# DataLoader 정의 (배치 처리, 패딩 추가)
def collate_batch(batch):
    text_list, label_list = zip(*batch)
    text_pad = torch.nn.utils.rnn.pad_sequence([torch.tensor(text) for text in text_list], batch_first=True, padding_value=vocab["<PAD>"])
    label_tensor = torch.tensor(label_list, dtype=torch.long)  # 다중 분류이므로 long 타입
    return text_pad, label_tensor

# DataLoader 설정
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)

# 결과 확인
print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('테스트 샘플의 개수 : {}'.format(len(test_data)))
print(f'단어 집합의 크기 : {len(vocab)}')

# Model
class CNN(nn.Module):
  def __init__(self, vocab_size, embedding_dim, n_kernels, kernel_sizes, output_dim, dropout, pad_idx):
    super().__init__()
    self.embedding = nn.Embedding(num_embeddings = vocab_size, # 임베딩을 할 단어들의 개수 (단어 집합의 크기)
                                  embedding_dim = embedding_dim, # 임베딩 할 벡터의 차원 (하이퍼파라미터)
                                  padding_idx = pad_idx) # 패딩을 위한 토큰의 인덱스
    self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, # input channel수 ( ex RGB 이미지 = 3 )
                                          out_channels = n_kernels, # convolution에 의해 생성될 channel의 수
                                          kernel_size = (ksize, embedding_dim)) # ksize만 변화. embedding_dim은 고정
                                for ksize in kernel_sizes])
    self.fc = nn.Linear(len(kernel_sizes)*n_kernels, output_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, review):
    embedded = self.embedding(review)
    embedded = embedded.unsqueeze(1) # 특정 위치에 1인 차원을 추가 <-> squeeze : 1인 차원을 제거
    conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
    pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
    cat = self.dropout(torch.cat(pooled, dim = 1))
    res = self.fc(cat)
    return self.fc(cat)


# 모델 선언
INPUT_DIM = len(vocab)
EMBEDDING_DIM = 300
N_KERNELS = 100
KERNEL_SIZES = [3,4,5]
OUTPUT_DIM = 7
DROPOUT = 0.5

PAD_IDX = vocab['<PAD>']
UNK_IDX = vocab['<UNK>']

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_KERNELS, KERNEL_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

print('모델 파라미터 수 :', sum(param.numel() for param in model.parameters() if param.requires_grad))
# 사전 훈련된 단어 벡터 불러오기
pretrained_weight = torch.randn(len(vocab), EMBEDDING_DIM)
print(pretrained_weight.shape, model.embedding.weight.data.shape)
print(f"사전 훈련된 임베딩 복사: {model.embedding.weight.data.copy_(pretrained_weight)}")

# unk, pad token -> 0 처리
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def categorical_accuracy(preds, y):
  # preds: [batch_size, num_classes], 모델의 출력 (예측 확률 분포)
  # y: [batch_size], 각 샘플에 대한 정수 클래스 레이블 (0 ~ num_classes-1)
    
  # softmax를 사용하여 예측된 각 클래스의 확률 중 가장 높은 인덱스를 선택
  _, predicted_labels = torch.max(preds, 1)  # pred: [batch_size, num_classes] -> predicted_labels: [batch_size]

  # 예측된 클래스와 실제 레이블 비교
  correct = (predicted_labels == y).float()  # 예측과 실제 레이블이 일치하면 1, 아니면 0
  acc = correct.sum() / len(correct)  # 정확도 = 맞춘 비율

  return acc
  
def train(model, iterator, optimizer, criterion):
  epoch_loss = 0
  epoch_acc = 0

  model.train()

  for batch in iterator:
    optimizer.zero_grad()
    predictions = model(batch[0]).squeeze(1) # output_dim = 1
    loss = criterion(predictions, batch[1])
    acc = categorical_accuracy(predictions, batch[1])

    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    epoch_acc += acc.item()

  return epoch_loss / len(iterator), epoch_acc / len(iterator)
  
def evaluate(model, iterator, criterion):
  epoch_loss = 0
  epoch_acc = 0

  model.eval()

  with torch.no_grad():
    for batch in iterator:
      predictions = model(batch[0]).squeeze(1)
      loss = criterion(predictions, batch[1])
      acc = categorical_accuracy(predictions, batch[1])

      epoch_loss += loss.item()
      epoch_acc += acc.item()

  return epoch_loss / len(iterator), epoch_acc / len(iterator)
  
def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs
  
N_EPOCHS = 5

model.to(device)

for epoch in range(N_EPOCHS):
  start_time = time.time()

  train_loss, train_acc = train(model, train_loader, optimizer, criterion)

  end_time = time.time()

  epoch_mins, epoch_secs = epoch_time(start_time, end_time)

  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    
model.load_state_dict(torch.load('MovieSentimentAnalysis.pt'))

test_loss, test_acc = evaluate(model, test_loader, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
=======
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from torchtext import datasets
from torchtext.vocab import build_vocab_from_iterator
import random
import numpy as np
import pandas as pd
from collections import Counter
from typing import List
from kiwipiepy import Kiwi
from sklearn.model_selection import train_test_split

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'사용할 디바이스: {device}')

KERNEL_SIZE = [3, 4, 5]
MAX_VOCAB_SIZE = 5000
SEED = 42
BATCH_SIZE = 32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

kiwi = Kiwi()

def most_frequent_number(lst):
    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0][0]
    
df = pd.read_excel('data/dataset.xlsx')
token_count = []
for i in range(len(df)):
  token_count.append(len(kiwi.tokenize(df['Sentence'][i], normalize_coda=False, split_complex=False, blocklist=None)))
print(f"데이터 1문장 평균 형태소 갯수: {most_frequent_number(token_count)}")


def postprocess_morphemes(morphemes):
    # "안녕하세요" -> "안녕하세요" (하 +세요 -> 안녕하세요)
    morphemes = merge_greetings(morphemes)
    
    return morphemes

def merge_greetings(morphemes):
    """인사말 '안녕 하세요'와 같은 부분을 합침"""
    merged = []
    i = 0
    while i < len(morphemes):
        # '안녕' + '하' + '세요' -> '안녕하세요'
        if morphemes[i] == '안녕' and i + 2 < len(morphemes) and morphemes[i + 1] == '하' and morphemes[i + 2] == '세요':
            merged.append('안녕하세요')
            i += 3
        else:
            merged.append(morphemes[i])
            i += 1
    return merged

def tokenizer(text):
  # 형태소로 나누기
  token_tmp = kiwi.tokenize(text, normalize_coda=False, split_complex=False, blocklist=None)
  token = []
  for t in token_tmp:
    token.append(t.form)
  token = postprocess_morphemes(token)
  
  # 패딩 토큰 추가
  if len(token) < max(KERNEL_SIZE):
    for i in range(0, max(KERNEL_SIZE)-len(token)):
      token.append('<PAD>') # 커널 사이즈보다 문장의 길이가 작은 경우 에러 방지
  return token

emo = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
emotion_to_label = {
  '공포': 0,
  '놀람': 1,
  '분노': 2,
  '슬픔': 3,
  '중립': 4,
  '행복': 5,
  '혐오': 6
}
emo_count = {0: 5468, 1: 5898, 2: 5665, 3: 5267, 4: 4830, 5: 6037, 6: 5429}

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

print(f"train_data: {len(train_df)}")
print(f"test_data: {len(test_df)}")
print(f"test_data_head: {test_df.head()}")


# class Vocab:
#   def __init__(self, tokens_list, min_freq=1):
#     counter = Counter()
#     for tokens in tokens_list:
#       counter.update(tokens)

#       # Special tokens 먼저 추가
#       self.itos = ['<PAD>', '<UNK>']
#       self.stoi = {'<PAD>': 0, '<UNK>': 1}

#       for token, freq in counter.items():
#         if freq >= min_freq and token not in self.stoi:
#           self.stoi[token] = len(self.itos)
#           self.itos.append(token)

#   def encode(self, tokens):
#     return [self.stoi.get(tok, self.stoi['<UNK>']) for tok in tokens]

#   def decode(self, indices):
#     return [self.itos[i] for i in indices]

# class TextDataset(Dataset):
#   def __init__(self, df, vocab, max_len=50):
#     self.texts = df['Sentence'].apply(tokenizer).tolist()  # 텍스트 토큰화
#     self.labels = df['Emotion'].tolist()  # 레이블
#     self.vocab = vocab  # Vocab 객체
#     self.max_len = max_len  # 문장의 최대 길이

#   def __len__(self):
#     return len(self.texts)

#   def __getitem__(self, idx):
#     token_ids = self.vocab.encode(self.texts[idx])  # 토큰 ID로 변환
#     tensor = torch.tensor(token_ids[:self.max_len])  # 최대 길이로 자르고 텐서로 변환
#     label = torch.tensor(self.labels[idx], dtype=torch.float)  # 레이블 텐서
#     return tensor, label

# def collate_fn(batch):
#   texts, labels = zip(*batch)
#   texts = pad_sequence(texts, batch_first=True, padding_value=0)
#   labels = torch.stack(labels)
#   return texts, labels

# tokenized_sentences = df['Sentence'].apply(tokenizer).tolist()
# vocab = Vocab(tokenized_sentences)

# lengths = [len(tokens) for tokens in tokenized_sentences]
# print(f"데이터 1문장 평균 토큰 갯수: {sum(lengths) / len(lengths)}")

# train_dataset = TextDataset(train_data, vocab)
# test_dataset = TextDataset(test_data, vocab)

# 텍스트와 레이블을 나누어 훈련 데이터셋과 테스트 데이터셋으로 사용
train_data = list(zip(train_df['Sentence'], train_df['Emotion']))
test_data = list(zip(test_df['Sentence'], test_df['Emotion']))

# 단어 집합 구축을 위한 토큰화 함수
def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)  # tokenizer를 사용하여 토큰화
        
# train_data로 단어 집합 생성 (unk_token 명시적으로 설정)
vocab = build_vocab_from_iterator(
    yield_tokens(train_data),
    max_tokens=MAX_VOCAB_SIZE,
    specials=["<PAD>", "<UNK>"],  # <unk>를 special token으로 설정
)

# <UNK>를 기본적으로 처리할 토큰으로 설정
vocab.set_default_index(vocab["<UNK>"])

# 텍스트를 인덱스로 변환하는 파이프라인
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]  # <UNK> 처리

# 레이블을 숫자로 변환하는 파이프라인 (7개의 클래스)
def label_pipeline(label):
    return emotion_to_label[label]  # 감정 이름을 숫자로 변환

# 데이터셋을 파이프라인 처리하여 텍스트와 레이블을 처리
train_data = [(text_pipeline(text), label_pipeline(label)) for text, label in train_data]
test_data = [(text_pipeline(text), label_pipeline(label)) for text, label in test_data]

# DataLoader 정의 (배치 처리, 패딩 추가)
def collate_batch(batch):
    text_list, label_list = zip(*batch)
    text_pad = torch.nn.utils.rnn.pad_sequence([torch.tensor(text) for text in text_list], batch_first=True, padding_value=vocab["<PAD>"])
    label_tensor = torch.tensor(label_list, dtype=torch.long)  # 다중 분류이므로 long 타입
    return text_pad, label_tensor

# DataLoader 설정
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)

# 결과 확인
print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('테스트 샘플의 개수 : {}'.format(len(test_data)))
print(f'단어 집합의 크기 : {len(vocab)}')

# Model
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


# 모델 선언
INPUT_DIM = len(vocab)
EMBEDDING_DIM = 512
N_KERNELS = 256
KERNEL_SIZES = [3,4,5]
OUTPUT_DIM = 7
DROPOUT = 0.5

PAD_IDX = vocab['<PAD>']
UNK_IDX = vocab['<UNK>']

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_KERNELS, KERNEL_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
model.to(device)

print('모델 파라미터 수 :', sum(param.numel() for param in model.parameters() if param.requires_grad))
# 사전 훈련된 단어 벡터 불러오기
pretrained_weight = torch.randn(len(vocab), EMBEDDING_DIM)
print(pretrained_weight.shape, model.embedding.weight.data.shape)
print(f"사전 훈련된 임베딩 복사: {model.embedding.weight.data.copy_(pretrained_weight)}")

# unk, pad token -> 0 처리
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
criterion = nn.CrossEntropyLoss()

def categorical_accuracy(preds, y):
  # preds: [batch_size, num_classes], 모델의 출력 (예측 확률 분포)
  # y: [batch_size], 각 샘플에 대한 정수 클래스 레이블 (0 ~ num_classes-1)
    
  # softmax를 사용하여 예측된 각 클래스의 확률 중 가장 높은 인덱스를 선택
  _, predicted_labels = torch.max(preds, 1)  # pred: [batch_size, num_classes] -> predicted_labels: [batch_size]

  # 예측된 클래스와 실제 레이블 비교
  correct = (predicted_labels == y).float()  # 예측과 실제 레이블이 일치하면 1, 아니면 0
  acc = correct.sum() / len(correct)  # 정확도 = 맞춘 비율

  return acc
  
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        
        text, labels = batch
        text = text.to(device)
        labels = labels.to(device)

        predictions = model(text)
        
        loss = criterion(predictions, labels)
        acc = categorical_accuracy(predictions, labels)
        
        loss.backward()
        
        # 그래디언트 클리핑 추가
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
  
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text = text.to(device)
            labels = labels.to(device)

            predictions = model(text)
            
            loss = criterion(predictions, labels)
            acc = categorical_accuracy(predictions, labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
  
def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs
  
N_EPOCHS = 30
best_valid_acc = 0

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_loader, criterion)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # 학습률 스케줄러 업데이트
    scheduler.step(valid_acc)
    
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), 'best-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

torch.save(vocab, 'vocab.pth')

# test_data를 이용해서 모델 평가

# model.load_state_dict(torch.load('best-model.pt'))
# model.eval()
# test_loss, test_acc = evaluate(model, test_loader, criterion, device)
# print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
>>>>>>> origin/main
