python .\main_train.py --num_epochs 3 --name "DFADD" --upsample_num 35785 --downsample_num 0
python .\main_train.py --num_epochs 3 --name "ASVspoof2021_DF" --upsample_num 104014 --downsample_num 235146
python .\main_train.py --num_epochs 3 --name "CodecFake" --upsample_num 104014 --downsample_num 265916 
python .\main_train_DCA.py --num_epochs 3 --name "DCA"

python .\generate_score_new.py --model_folder "./models/DFADD" --task "DFADD"
python .\generate_score_new.py --model_folder "./models/ASVspoof2021_DF" --task "ASVspoof2021_DF"
python .\generate_score_new.py --model_folder "./models/CodecFake" --task "CodecFake"
python .\generate_score_new.py --model_folder "./models/DCA" --task "DFADD"
python .\generate_score_new.py --model_folder "./models/DCA" --task "ASVspoof2021_DF"
python .\generate_score_new.py --model_folder "./models/DCA" --task "CodecFake"
python .\generate_score_new.py --model_folder "./models/DCA" --task "in_the_wild"
python .\generate_score_new.py --model_folder "./models/in_the_wild" --task "in_the_wild"
