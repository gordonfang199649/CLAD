from main_train import *
import random
def test_model_pipeline(model, queue, device, database_path, cut_length=64600):
    """
    測試模型和後續邏輯是否能正常運行，避免後期報錯。

    Args:
        model (nn.Module): 模型對象
        queue (NegativeQueue): 負樣本隊列對象
        feature_dim (int): 嵌入特徵的維度
        device (torch.device): 運行設備

    Returns:
        bool: 測試是否成功
    """
    print("開始測試模型流程...")

    try:
        # 1. 模擬輸入數據
        d_label_trn, file_dev, utt2spk = genSpoof_list(
            dir_meta=database_path + "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", 
            is_train=False, is_eval=False
        )
        dev_dataset = Dataset_ASVspoof2019_train(
            list_IDs=file_dev,
            labels=d_label_trn,
            base_dir=os.path.join(database_path + 'ASVspoof2019_LA_eval/'),
            cut_length=cut_length,
            utt2spk=utt2spk
        )
        selected_indices = random.sample(range(len(dev_dataset)), 1000)
        subset =  Subset(dev_dataset, selected_indices)
        validation_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=1)
        validation_flow = iter(validation_loader)
        for i in trange(0, len(validation_loader), total=len(validation_loader), initial=0):
            audio_input, spks, labels = next(validation_flow)
            # 2. 模型前向傳播
            audio_input = preprocess_audio(audio_input.squeeze(1).to(device))
            features = model.encoder(audio_input)  # 獲取嵌入特徵

            # 3. 對比損失計算
            negatives = queue.get_negatives()
            loss_cl = contrastive_loss(features, features, negatives, temperature=0.07)
            print(f'對比損失計算{loss_cl}')
            assert isinstance(loss_cl, torch.Tensor), "對比損失計算錯誤"

            # 4. 長度損失計算
            loss_len = length_loss(features, labels, margin=4.0)
            print(f'對比損失計算{loss_len}')
            assert isinstance(loss_len, torch.Tensor), "長度損失計算錯誤"

            # 5. 更新負樣本隊列
            queue.dequeue_and_enqueue(features, labels)
            print("測試流程完成，未發現問題。")
        return True

    except Exception as e:
        print(f"測試失敗，錯誤信息：{e}")
        return False

def test_validate_function(model, queue, feature_dim, device, args, database_path, cut_length=64600):
    print("開始測試 validate 函數...")

    try:
        # 1. 加載部分驗證數據集
        d_label_trn, file_dev, utt2spk = genSpoof_list(
            dir_meta=database_path + "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", 
            is_train=False, is_eval=False
        )
        dev_dataset = Dataset_ASVspoof2019_train(
            list_IDs=file_dev,  # 只選取 batch_size 的樣本
            labels=d_label_trn,
            base_dir=os.path.join(database_path + 'ASVspoof2019_LA_eval/'),
            cut_length=cut_length,
            utt2spk=utt2spk
        )
        validation_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)

        # 3. 執行 validate 函數
        total_loss = validate(model, validation_loader, device, queue, epoch_num=1, args=args)

        # 4. 驗證結果
        print(f"測試 validate 函數成功，總損失: {total_loss:.4f}")
        return True

    except Exception as e:
        print(f"測試 validate 函數失敗，錯誤信息：{e}")
        return False

if __name__ == "__main__":
    args = initParams()
    cuda = torch.cuda.is_available()
    print('Cuda device available: ', cuda)
    device = torch.device("cuda:0" if cuda else "cpu")
    
    with open("./config.conf", "r") as f_json:
        config = json.loads(f_json.read())
        
    with open(config['aasist_config_path'], "r") as f_json:
        aasist_config = json.loads(f_json.read())
    # aasist_model_config = aasist_config["model_config"]
    # aasist_encoder = AASIST.AasistEncoder(aasist_model_config).to(device)
    # downstream_model = DownStreamLinearClassifier(aasist_encoder, input_depth=160)
    # model = downstream_model.to(device)
    
    model = torch.load('./models/checkpoint/anti-spoofing_feat_model_5.pt').to(device)
    
    queue = NegativeQueue(feature_dim=160, queue_size=args.queue_size)
    database_path = config['database_path']
    # test_model_pipeline(model=model, queue=queue, device=device, database_path=database_path)
    test_validate_function(model=model, queue=queue, feature_dim=64600, device=device, database_path=database_path, args=args)