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
from TrainingUtils import NegativeQueue, MomentumUpdater

"""
    給 CommandLine 下的超參數與指令

    Returns:
        argparse.Namespace: Parsed arguments.
"""
def initParams() -> argparse.Namespace:
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

"""
    計算對比式學習 Loss

    Args:
        features_q (torch.Tensor): 模型萃取人聲表徵
        features_k (torch.Tensor): 模型萃取人聲表徵
        negatives (torch.Tensor): 負向樣本
        temperature (float): temperature
    Returns:
        torch.Tensor: Length loss.
"""
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

"""
    計算真假人聲長度 Loss

    Args:
        features (torch.Tensor): 模型萃取人聲表徵
        labels (torch.Tensor): 真假人聲標籤
        margin (float): Margin for the loss calculation.

    Returns:
        torch.Tensor: Length loss.
"""
def length_loss(features: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    real_features = features[labels == 0]
    fake_features = features[labels == 1]
    print(f'真實人聲大小: {real_features.shape}')
    print(f'合成人聲大小: {fake_features.shape}')
    real_loss = torch.norm(real_features, p=2, dim=1).mean()
    fake_loss = torch.relu(margin - torch.norm(fake_features, p=2, dim=1)).mean()
    print(f'真實人聲 Loss: {real_loss}')
    print(f'合成人聲 Loss: {fake_loss}')
    return real_loss + fake_loss

"""
    執行驗證集

    Args:
        model (nn.Module): Detection 模型
        validation_loader (DataLoader): 驗證集資料容器
        device (torch.device): GPU 設備
        queue (NegativeQueue): Negative sample queue.
        epoch_num (int): Current epoch number.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        eer 驗證集計算下來的 EER
        average_loss 平均 Loss
"""
def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    device: torch.device,
    queue: NegativeQueue,
    epoch_num: int,
    args
) -> float:
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
            # 音頻前處理交給 RawAudio 類去實作
            # audio_input = preprocess_audio(audio_input.squeeze(1).to(device))
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
        
        with open(os.path.join(args.output_path, args.name, "validation_loss.log"), "a") as log:
            log.write(f"epoch: {epoch_num}\t EER: {eer}\t FRR: {frr}\t FAR: {far}\t Threshold: {threshold}\t Loss: {average_loss}\n")
        print(f"Validation EER: {eer:.4f}")

    return eer, average_loss

"""
    前處理音頻
    1. 取樣 64600 個採樣點，通常音頻是 16kHz， 64600/16000 大約是 4 秒時間
    2. 音頻不足 4 秒則重複片段填充
    3. 音頻超過 4 秒則截斷超過片段
    
    Args:
    audio_input (torch.Tensor): 聲音 raw data
    target_length (int): 採樣數
    
    Returns:
    處理過後的音頻片段
"""
def preprocess_audio(audio_input: torch.Tensor, target_length: int = 64600) -> torch.Tensor:
    if audio_input.shape[-1] < target_length:
        repeat_factor = int(target_length / audio_input.shape[-1]) + 1
        audio_input = audio_input.repeat(1, repeat_factor)[:, :target_length]
    elif audio_input.shape[-1] > target_length:
        audio_input = audio_input[:, :target_length]
    return audio_input

"""
    執行訓練集

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        device (torch.device): GPU 設備

    Returns:

"""
def train(args, device)->None:
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
    
    # 訓練集
    training_set = RawAudio(path_to_database=f'../datasets/{args.name}'
                            , meta_csv = 'meta.csv'
                            , nb_samp=args.nb_samp
                            , return_label=True
                            , part = 'train')
    
    # 分離真實人聲和假人聲
    meta = pd.read_csv(f'../datasets/{args.name}/train/meta.csv')
    real_indices = meta[meta['label'] == 'bonafide'].index.tolist()
    spoof_indices = meta[meta['label'] == 'spoof'].index.tolist()
    
    # 如果需要下採樣假人聲
    if args.downsample_num > 0 and len(spoof_indices) > args.downsample_num:
        selected_spoof_indices = random.sample(spoof_indices, args.downsample_num)
    else:
        selected_spoof_indices = spoof_indices  # 如果不需要下採樣，保留所有假人聲

    # 新的數據集(下採樣假人聲)
    real_subset = Subset(training_set, real_indices)
    spoof_subset = Subset(training_set, selected_spoof_indices)
    training_set = ConcatDataset([real_subset, spoof_subset])
    
    # 如果需要從 LibriSpeech 上採樣真實人聲
    if args.upsample_num > 0:
        training_set_real_utterance =  RawAudio(path_to_database=f'../datasets/LibriSpeech'
                            , meta_csv = 'meta.csv'
                            , return_label=True
                            , nb_samp=args.nb_samp
                            , part = 'train')
        selected_bonafide_indices = random.sample(range(len(training_set_real_utterance)), args.upsample_num)
        bonafide_subset =  Subset(training_set_real_utterance, selected_bonafide_indices)
        training_set = ConcatDataset([training_set, bonafide_subset])
        
    train_loader = DataLoader(training_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=args.nb_worker)
    train_flow = iter(train_loader)
    
    # 驗證集
    validation_set = RawAudio(path_to_database=f'../datasets/{args.name}'
                            , meta_csv = 'meta.csv'
                            , return_label=True
                            , nb_samp=args.nb_samp
                            , part = 'validation')
    
    validation_loader = DataLoader(validation_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=args.nb_worker)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_annealing_epochs)

    best_val_eer= float('inf')
    best_model_path = os.path.join(args.output_path, args.name, "best_model.pth")

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
            # 音頻前處理交給 RawAudio 類去實作
            # audio_input = preprocess_audio(audio_input.squeeze(1).to(device))
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

"""
    主程式入口點
"""
if __name__ == "__main__":
    args = initParams()

    # 檢查可用 CUDA 設備
    cuda = torch.cuda.is_available()
    print('Cuda device available: ', cuda)
    device = torch.device("cuda" if cuda else "cpu")

    # 印出當前 CUDA_VISIBLE_DEVICES 的設備
    if cuda:
        print(f"Using device: {torch.cuda.current_device()}, Name: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")

    # 啟動訓練
    train(args, device)
