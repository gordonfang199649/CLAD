import torch
import torch.nn as nn
import argparse
import os
import json
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.nn.functional import softmax
from tqdm import trange
import AASIST
from Model import DownStreamLinearClassifier
from evaluate_tDCF_asvspoof19 import compute_eer_frr_far
import random
import pandas as pd
from dataloader import RawAudio

class NegativeQueue:
    """
    Implements a queue for storing negative samples for contrastive learning.
    """
    def __init__(self, feature_dim: int, queue_size: int):
        self.queue = torch.randn(queue_size, feature_dim).cuda()
        self.queue = self.queue / self.queue.norm(dim=1, keepdim=True)  # 歸一化
        self.ptr = queue_size  # 初始化時將 ptr 設置為 queue_size，視為已填滿
        self.queue_size = queue_size
        self.labels = torch.ones(queue_size).cuda()  # 初始化為負樣本標籤

    def dequeue_and_enqueue(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        negative_features = features[labels == 1]  # 選取負樣本
        if negative_features.size(0) == 0:  # 如果沒有負樣本，直接返回
            return

        negative_features = negative_features / negative_features.norm(dim=1, keepdim=True)  # 歸一化
        batch_size = negative_features.size(0)

        if batch_size > self.queue_size:
            self.queue = negative_features[-self.queue_size:].detach()
            self.ptr = 0
        else:
            end_ptr = (self.ptr + batch_size) % self.queue_size
            if end_ptr < self.ptr:
                self.queue[self.ptr:] = negative_features[:self.queue_size - self.ptr].detach()
                self.queue[:end_ptr] = negative_features[self.queue_size - self.ptr:].detach()
            else:
                self.queue[self.ptr:end_ptr] = negative_features.detach()
            self.ptr = end_ptr

    def get_negatives(self) -> torch.Tensor:
        """
        Returns negative samples from the queue.

        Returns:
            torch.Tensor: Negative samples.
        """
        return self.queue


class MomentumUpdater:
    """
    Updates the parameters of a momentum-based encoder.
    """
    def __init__(self, model: nn.Module, momentum: float = 0.999):
        self.model = model
        self.momentum = momentum
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self, new_model: nn.Module) -> None:
        """
        Updates the parameters of the shadow model using momentum.

        Args:
            new_model (nn.Module): The model with updated parameters.
        """
        with torch.no_grad():
            for name, param in new_model.named_parameters():
                if name in self.shadow:
                    self.shadow[name] = self.momentum * self.shadow[name] + (1 - self.momentum) * param.data
                    param.data = self.shadow[name]


def initParams() -> argparse.Namespace:
    """
    Initializes and parses command-line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CLAD Model Training")
    parser.add_argument('--name', type = str, required = True)
    parser.add_argument('--upsample_num', type=int, help="real utterance augmentation number", default=0)
    parser.add_argument('--downsample_num', type=int, help="fake utterance augmentation number", default=0)
    parser.add_argument('-nb_worker', type = int, default = 8)
    parser.add_argument('--seed', type=int, help="Random seed", default=42)
    parser.add_argument('--output_path', type=str, help="Output folder", default='./models/')
    parser.add_argument('--model', type=str, choices=['encoder'], default='encoder', help="Model architecture")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=3, help="Number of epochs")
    parser.add_argument('--learning_rate', type=float, default=0.0005, help="Learning rate")
    parser.add_argument('--temperature', type=float, default=0.07, help="Temperature for contrastive loss")
    parser.add_argument('--margin', type=float, default=4.0, help="Margin for length loss")
    parser.add_argument('--queue_size', type=int, default=6144, help="Queue size for negative samples")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="Weight decay for optimizer")
    parser.add_argument('--cosine_annealing_epochs', type=int, default=3, help="Number of epochs for cosine annealing")
    parser.add_argument('--length_loss_weight', type=float, default=2.0, help="Weight for length loss in final loss computation")
    parser.add_argument('--cross_entropy_weight', type=float, default=1.0, help="Weight for cross-entropy loss in downstream task")
    parser.add_argument('-nb_samp', type = int, default = 64600)
    args = parser.parse_args()
    return args

def contrastive_loss(features_q: torch.Tensor, features_k: torch.Tensor, negatives: torch.Tensor, temperature: float) -> torch.Tensor:
    features_q = features_q / features_q.norm(dim=1, keepdim=True)
    features_k = features_k / features_k.norm(dim=1, keepdim=True)
    negatives = negatives / negatives.norm(dim=1, keepdim=True)

    pos_sim = torch.exp(torch.sum(features_q * features_k, dim=-1) / temperature)
    neg_sim = torch.exp(torch.matmul(features_q, negatives.T) / temperature)

    # 避免數值不穩定
    epsilon = 1e-8  # 平滑項
    pos_sim = torch.clamp(pos_sim, min=epsilon)
    neg_sim_sum = torch.clamp(neg_sim.sum(dim=-1), min=epsilon)

    loss = -torch.log(pos_sim / (pos_sim + neg_sim_sum))
    return loss.mean()


def length_loss(features: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """
    Computes length loss for real and fake features.

    Args:
        features (torch.Tensor): Features of audio samples.
        labels (torch.Tensor): Corresponding labels for the features.
        margin (float): Margin for the loss calculation.

    Returns:
        torch.Tensor: Length loss.
    """
    real_features = features[labels == 0]
    fake_features = features[labels == 1]
    print(f'真實人聲大小: {real_features.shape}')
    print(f'合成人聲大小: {fake_features.shape}')
    real_loss = torch.norm(real_features, p=2, dim=1).mean()
    fake_loss = torch.relu(margin - torch.norm(fake_features, p=2, dim=1)).mean()
    print(f'真實人聲 Loss: {real_loss}')
    print(f'合成人聲 Loss: {fake_loss}')
    return real_loss + fake_loss

def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    device: torch.device,
    queue: NegativeQueue,
    epoch_num: int,
    args
) -> float:
    """
    Validates the model on the validation dataset.

    Args:
        model (nn.Module): The trained model.
        validation_loader (DataLoader): Validation dataset loader.
        device (torch.device): Device for computation.
        queue (NegativeQueue): Negative sample queue.
        epoch_num (int): Current epoch number.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    cross_entropy_criterion = nn.CrossEntropyLoss()
    score_loader = []
    label_loader = []

    validation_flow = iter(validation_loader)
    
    with torch.no_grad():
        for i in trange(0, len(validation_loader), total=len(validation_loader), initial=0):
            try:
                audio_input, labels = next(validation_flow)
            except StopIteration:
                validation_flow = iter(validation_loader)
                audio_input, labels = next(validation_flow)
                
            #audio_input = preprocess_audio(audio_input.squeeze(1).to(device))
            audio_input = audio_input.to(device)
            labels = labels.to(device)
            features = model.encoder(audio_input)
            logits = model(audio_input)

            # 負樣本隊列
            negatives = queue.get_negatives()

            # 損失計算
            loss_cl = contrastive_loss(features, features, negatives, temperature=args.temperature)
            loss_len = length_loss(features, labels, margin=4.0)
            loss_ce = cross_entropy_criterion(logits, labels)
            
            print(f"驗證集對比式學習 Loss: {loss_cl}")
            print(f"驗證集長度 Loss: {loss_len}")
            print(f"驗證集 Cross Entropy Loss: {loss_ce}")

            # 總損失計算
            # 預防 Loss NaN的問題，順便紀錄哪一個 Loss 出錯
            loss = 0
            if not torch.isnan(loss_cl):
                loss += loss_cl
            else:
                print("Contrastive loss is NaN, ignoring this term.")

            if not torch.isnan(loss_len):
                loss += args.length_loss_weight * loss_len
            else:
                print("Length loss is NaN, ignoring this term.")

            if not torch.isnan(loss_ce):
                loss += args.cross_entropy_weight * loss_ce
            else:
                print("Cross-entropy loss is NaN, ignoring this term.")
            total_loss += loss
            total_samples += audio_input.size(0)

            queue.dequeue_and_enqueue(features, labels)
            
            # 儲存分數和標籤
            scores = softmax(logits, dim=1)[:, 0]
            score_loader.append(scores)
            label_loader.append(labels)

        average_loss = total_loss / total_samples
        print(f"驗證集 Cross Total Loss: {average_loss}")

        # 計算 EER
        scores = torch.cat(score_loader, 0).data.cpu().numpy()
        labels = torch.cat(label_loader, 0).data.cpu().numpy()
        eer, frr, far, threshold = compute_eer_frr_far(scores[labels == 0], scores[labels == 1])

        # 儲存日誌
        save_path = os.path.join(args.output_path, args.name)
        os.makedirs(save_path, exist_ok = True)
        
        with open(os.path.join(args.output_path, "dev_loss.log"), "a") as log:
            log.write(f"epoch: {epoch_num}\t EER: {eer}\t FRR: {frr}\t FAR: {far}\t Threshold: {threshold}\t Loss: {average_loss}\n")
        print(f"Validation EER: {eer:.4f}")

    return eer, average_loss


def preprocess_audio(audio_input: torch.Tensor, target_length: int = 64600) -> torch.Tensor:
    if audio_input.shape[-1] < target_length:
        repeat_factor = int(target_length / audio_input.shape[-1]) + 1
        audio_input = audio_input.repeat(1, repeat_factor)[:, :target_length]
    elif audio_input.shape[-1] > target_length:
        audio_input = audio_input[:, :target_length]
    return audio_input

def train(args, device):
    torch.manual_seed(args.seed)
    with open("./config.conf", "r") as f_json:
        config = json.loads(f_json.read())

    with open(config['aasist_config_path'], "r") as f_json:
        aasist_config = json.loads(f_json.read())
    aasist_model_config = aasist_config["model_config"]
    aasist_encoder = AASIST.AasistEncoder(aasist_model_config).to(device)
    downstream_model = DownStreamLinearClassifier(aasist_encoder, input_depth=160)
    model = downstream_model.to(device)
    momentum_updater = MomentumUpdater(model.encoder)
    
    # 加載三個主要數據集
    training_sets = {
        "DFADD": RawAudio(path_to_database=f'../datasets/DFADD'
                            , meta_csv = 'meta.csv'
                            , return_label=True, nb_samp=args.nb_samp
                            , part = 'train'),
        "CodecFake": RawAudio(path_to_database=f'../datasets/CodecFake'
                            , meta_csv = 'meta.csv'
                            , return_label=True, nb_samp=args.nb_samp
                            , part = 'train'),
        "ASVspoof2021_DF": RawAudio(path_to_database=f'../datasets/ASVspoof2021_DF'
                            , meta_csv = 'meta.csv'
                            , return_label=True, nb_samp=args.nb_samp
                            , part = 'train')
    }

    # 定義下採樣目標數量
    TARGET_FAKE_COUNT = 126321

    # 初始化訓練集列表
    training_set_list = []

    # 遍歷每個數據集
    for name, training_set in training_sets.items():
        # 加載 meta 信息
        meta = pd.read_csv(f'../datasets/{name}/train/meta.csv')
        
        # 獲取真實和假人聲的索引
        real_indices = meta[meta['label'] == 'bonafide'].index.tolist()
        spoof_indices = meta[meta['label'] == 'spoof'].index.tolist()

        # 下採樣假人聲
        if len(spoof_indices) > TARGET_FAKE_COUNT:
            selected_spoof_indices = random.sample(spoof_indices, TARGET_FAKE_COUNT)
        else:
            selected_spoof_indices = spoof_indices

        # 創建子數據集
        real_subset = Subset(training_set, real_indices)
        spoof_subset = Subset(training_set, selected_spoof_indices)
        
        # 合併真實和假樣本
        adjusted_set = ConcatDataset([real_subset, spoof_subset])
        training_set_list.append(adjusted_set)

    # 加載 LibriSpeech 數據集
    librispeech_set = RawAudio(path_to_database=f'../datasets/LibriSpeech'
                            , meta_csv = 'meta.csv'
                            , return_label=True, nb_samp=args.nb_samp
                            , part = 'train')
    training_set_list.append(librispeech_set)

    # 合併所有數據集
    final_training_set = ConcatDataset(training_set_list)

    # 創建 DataLoader
    train_loader = DataLoader(final_training_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=args.nb_worker)
    train_flow = iter(train_loader)

    # 驗證集處理
    validation_set_list = [RawAudio(path_to_database=f'../datasets/DFADD'
                            , meta_csv = 'meta.csv'
                            , return_label=True, nb_samp=args.nb_samp
                            , part = 'validation')
                        ,RawAudio(path_to_database=f'../datasets/CodecFake'
                            , meta_csv = 'meta.csv'
                            , return_label=True, nb_samp=args.nb_samp
                            , part = 'validation')
                        , RawAudio(path_to_database=f'../datasets/ASVspoof2021_DF'
                            , meta_csv = 'meta.csv'
                            , return_label=True, nb_samp=args.nb_samp
                            , part = 'validation')]
    # 合併所有數據集
    validation_set = ConcatDataset(validation_set_list)
    validation_loader = DataLoader(validation_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=args.nb_worker)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_annealing_epochs)

    best_val_eer= float('inf')
    best_model_path = os.path.join(args.output_path, "best_model.pth")

    queue = NegativeQueue(feature_dim=160, queue_size=args.queue_size)

    cross_entropy_criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for i in trange(0, len(train_loader), total=len(train_loader), initial=0):
            try:
                audio_input, labels = next(train_flow)
            except StopIteration:
                train_flow = iter(train_loader)
                audio_input, labels = next(train_flow)
            
            #audio_input = preprocess_audio(audio_input.squeeze(1).to(device))
            audio_input = audio_input.to(device)
            labels = labels.to(device)
            features = model.encoder(audio_input)
            logits = model(audio_input)

            negatives = queue.get_negatives()

            loss_cl = contrastive_loss(features, features, negatives, temperature=args.temperature)
            loss_len = length_loss(features, labels, margin=args.margin)
            loss_ce = cross_entropy_criterion(logits, labels)
            
            print(f"訓練集對比式學習 Loss: {loss_cl}")
            print(f"訓練集長度 Loss: {loss_len}")
            print(f"訓練集 Cross Entropy Loss: {loss_ce}")

            # 預防 Loss NaN的問題，順便紀錄哪一個 Loss 出錯
            loss = 0
            if not torch.isnan(loss_cl):
                loss += loss_cl
            else:
                print("Contrastive loss is NaN, ignoring this term.")

            if not torch.isnan(loss_len):
                loss += args.length_loss_weight * loss_len
            else:
                print("Length loss is NaN, ignoring this term.")

            if not torch.isnan(loss_ce):
                loss += args.cross_entropy_weight * loss_ce
            else:
                print("Cross-entropy loss is NaN, ignoring this term.")

            print(f"訓練集 Total Loss: {loss}")
            if torch.isnan(loss):
                print(f"NaN loss encountered at step {i}, skipping update.")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            queue.dequeue_and_enqueue(features, labels)
            momentum_updater.update(model.encoder)

            total_loss += loss.item() * audio_input.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")

        val_eer, val_loss = validate(model, validation_loader, device, queue, epoch + 1, args)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

        scheduler.step()
        save_path = os.path.join(args.output_path, args.name, 'checkpoint')
        os.makedirs(save_path, exist_ok = True)
        torch.save(model, os.path.join(save_path,
                                        'anti-spoofing_feat_model_%d.pt' % (epoch+1)))
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            torch.save(model, best_model_path)
            print(f"Best model saved at epoch {epoch+1}")

if __name__ == "__main__":
    args = initParams()
    cuda = torch.cuda.is_available()
    print('Cuda device available: ', cuda)
    device = torch.device("cuda:0" if cuda else "cpu")
    train(args, device)
