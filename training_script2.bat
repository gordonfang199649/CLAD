
:: 訓練 ASVspoof2021_DF
set CUDA_VISIBLE_DEVICES=1 && python .\main_train.py --num_epochs 5 --name "ASVspoof2021_DF" --upsample_num 104014 --downsample_num 235146

:: 訓練 ASVspoof2021_DF
set CUDA_VISIBLE_DEVICES=1 && python .\main_train_DCA.py --num_epochs 5 --name "DCA"

:: 暫停視窗，等待用戶按任意鍵繼續
pause
