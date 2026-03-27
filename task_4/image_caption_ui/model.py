# model.py  –  all ML logic, no Streamlit dependency
# ─────────────────────────────────────────────────────────────────────────────
import os, pickle
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Vocabulary ────────────────────────────────────────────────────────────────
class Vocabulary:
    SPECIAL = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

    def __init__(self, freq_threshold: int = 5):
        self.itos = dict(self.SPECIAL)
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.stoi)

    def build_vocabulary(self, sentence_list: list[str]):
        frequencies: Counter = Counter()
        idx = len(self.SPECIAL)
        for sentence in sentence_list:
            for word in sentence.lower().split():
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text: str) -> list[int]:
        unk = self.stoi["<UNK>"]
        return [self.stoi.get(t, unk) for t in text.lower().split()]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Dataset ───────────────────────────────────────────────────────────────────
class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, captions_df, transform=None, vocab=None):
        self.root_dir  = root_dir
        self.df        = captions_df.reset_index(drop=True)
        self.transform = transform
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary(freq_threshold=5)
            self.vocab.build_vocabulary(self.df["caption"].tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.root_dir, row["image"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tokens = (
            [self.vocab.stoi["<SOS>"]]
            + self.vocab.numericalize(row["caption"])
            + [self.vocab.stoi["<EOS>"]]
        )
        return img, torch.tensor(tokens, dtype=torch.long)


class MyCollate:
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs    = torch.stack([b[0] for b in batch])
        targets = pad_sequence([b[1] for b in batch], batch_first=True,
                               padding_value=self.pad_idx)
        return imgs, targets


# ── Model ─────────────────────────────────────────────────────────────────────
class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int, train_CNN: bool = False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for p in resnet.parameters():
            p.requires_grad = train_CNN
        resnet.fc = nn.Linear(resnet.fc.in_features, embed_size)
        resnet.fc.weight.requires_grad_(True)
        resnet.fc.bias.requires_grad_(True)
        self.resnet  = resnet
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        return self.dropout(self.relu(self.resnet(images)))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int,
                 vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_size)
        self.lstm    = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear  = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        emb = self.dropout(self.embed(captions))
        emb = torch.cat((features.unsqueeze(1), emb), dim=1)
        h, _ = self.lstm(emb)
        return self.linear(h)


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        return self.decoderRNN(self.encoderCNN(images), captions)


# ── Beam Search ───────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_caption_beam_search(
    image,
    model: CNNtoRNN,
    vocab: Vocabulary,
    device,
    beam_size: int = 5,
    max_length: int = 50,
) -> list[str]:
    model.eval()
    if isinstance(image, Image.Image):
        img_t = transform(image).unsqueeze(0).to(device)
    else:
        img_t = (image.unsqueeze(0) if image.dim() == 3 else image).to(device)

    features = model.encoderCNN(img_t).unsqueeze(1)
    _, states = model.decoderRNN.lstm(features)

    sos, eos = vocab.stoi["<SOS>"], vocab.stoi["<EOS>"]
    beams: list[tuple] = [(0.0, [sos], states)]
    completed: list[tuple] = []

    for _ in range(max_length):
        new_beams = []
        for score, caption, last_states in beams:
            last_word = torch.tensor([[caption[-1]]], device=device)
            emb = model.decoderRNN.embed(last_word)
            out, curr = model.decoderRNN.lstm(emb, last_states)
            log_p = F.log_softmax(model.decoderRNN.linear(out.squeeze(1)), dim=-1)
            top_lp, top_idx = log_p.topk(beam_size, dim=-1)
            for k in range(beam_size):
                nw = top_idx[0, k].item()
                ns = score + top_lp[0, k].item()
                nc = caption + [nw]
                (completed if nw == eos else new_beams).append(
                    (ns, nc) if nw == eos else (ns, nc, curr)
                )
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        if not beams:
            break

    all_cands = completed + [(s, c) for s, c, _ in beams]
    _, best = max(all_cands, key=lambda x: x[0])
    return [vocab.itos[i] for i in best if vocab.itos[i] not in {"<SOS>", "<EOS>"}]


# ── Trainer ───────────────────────────────────────────────────────────────────
class CaptionTrainer:
    def __init__(self, model, train_dataset, val_dataset, device, vocab,
                 checkpoint_path="checkpoint.pth.tar"):
        self.model           = model.to(device)
        self.train_dataset   = train_dataset
        self.val_dataset     = val_dataset
        self.device          = device
        self.vocab           = vocab
        self.checkpoint_path = checkpoint_path
        self.pad_idx         = vocab.stoi["<PAD>"]
        self.criterion       = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.best_val_loss   = float("inf")
        self.start_epoch     = 0

    def _loss_on_batch(self, imgs, caps):
        out     = self.model(imgs, caps[:, :-1])
        logits  = out[:, 1:, :].reshape(-1, out.shape[2])
        targets = caps[:, 1:].reshape(-1)
        return self.criterion(logits, targets)

    def save_checkpoint(self, epoch, optimizer, is_best=False):
        state = {
            "epoch":         epoch + 1,
            "state_dict":    self.model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(state, self.checkpoint_path)
        if is_best:
            torch.save(state, "best_model.pth.tar")

    def load_checkpoint(self, optimizer) -> bool:
        if not os.path.exists(self.checkpoint_path):
            return False
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch   = ckpt["epoch"]
        self.best_val_loss = ckpt["best_val_loss"]
        return True

    def train_epoch(self, train_loader, optimizer) -> float:
        self.model.train()
        total = 0.0
        for imgs, caps in train_loader:
            imgs, caps = imgs.to(self.device), caps.to(self.device)
            loss = self._loss_on_batch(imgs, caps)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        return total / len(train_loader)

    def eval_epoch(self, loader) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for imgs, caps in loader:
                imgs, caps = imgs.to(self.device), caps.to(self.device)
                total += self._loss_on_batch(imgs, caps).item()
        return total / len(loader)

    def predict(self, image, beam_size=5, max_length=50) -> list[str]:
        return generate_caption_beam_search(
            image, self.model, self.vocab, self.device,
            beam_size=beam_size, max_length=max_length,
        )