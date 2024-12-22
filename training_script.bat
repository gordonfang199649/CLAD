:: 訓練 DFADD
set CUDA_VISIBLE_DEVICES=0 && python .\main_train.py --num_epochs 5 --name "DFADD" --upsample_num 35785 --downsample_num 0

:: 訓練 CodecFake
set CUDA_VISIBLE_DEVICES=0 && python .\main_train.py --num_epochs 5 --name "CodecFake" --upsample_num 104014 --downsample_num 265916

:: 暫停視窗，等待用戶按任意鍵繼續
pause
