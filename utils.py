# utils.py
import torch
import torch.nn as nn
import pickle
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('stopwords')

MAX_LEN = 300

# Define the model class (same as training)
class LSTMSentimentModel(nn.Module):
  def __init__(self,vocab_size):
    super().__init__()
    self.embedding=nn.Embedding(vocab_size,embedding_dim=128)
    self.dropout = nn.Dropout(0.3)  # Dropout after embedding
    self.lstm=nn.LSTM(input_size=128,hidden_size=128, num_layers=2,batch_first=True)
    self.fc=nn.Linear(128,1)


  def forward(self, x ):
    embedded=self.embedding(x)
    embedded = self.dropout(embedded)  # Apply dropout
    inter_ht_states,(final_ht,final_ct)=self.lstm(embedded)
    output=self.fc(final_ht[-1])
    return output

# Preprocess text
def preprocess(text, vocab):
    tokens = word_tokenize(text.lower())
    indices = [vocab.get(token, vocab.get("<unk>", 0)) for token in tokens]
    if len(indices) < MAX_LEN:
        indices += [vocab["<pad>"]] * (MAX_LEN - len(indices))
    else:
        indices = indices[:MAX_LEN]
    return torch.tensor(indices).unsqueeze(0)  # shape: (1, max_len)

# Load model and vocab
def load_model_and_vocab():
    with open("word_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    model = LSTMSentimentModel(len(vocab))
    model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
    model.eval()
    return model, vocab
