To add a new dataset

1.create dataset folder:hotel_mask_sing_asp_dataset

2. put review.csv + business.csv + processed + prceossed/subword_encoder




3. 建立一个新 data_loaders/hotel_mask_sing_asp_dataset.py
3.1 把class Hotel_Mask_Dataset换成 class Hotel_Mask_Sing_Asp_Dataset

3.2 然后把所有hotel_mask 替换成 hotel_mask_sing_asp

4. summ_dataset_factory.py里面添加一个条目

5. project_settings.py里面增加一个条目，并把里面的dataset_dir改掉

6.bash scripts/preprocess_data.sh hotel_mask_sing_asp_dataset.py

7.在checkpoints/下面的三个sum,clf,lm中各建立一个同名dataset文件夹

8. 在output/eval/下面建立一个文件夹