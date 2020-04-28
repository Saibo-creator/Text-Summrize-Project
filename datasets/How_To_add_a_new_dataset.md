To add a new dataset

1. 深呼吸！

2  **build_rev.json_asp0-5.py.**  (create dataset folder:hotel_mask_sing_asp_dataset，build  put review.csv + business.csv  into hotel_mask_sing_asp_dataset/)


3. **建立一个新 data_loaders/hotel_mask_sing_asp_dataset.py**
  3.1 把class Hotel_Mask_Dataset换成 class Hotel_Mask_Sing_Asp_Dataset

  3.2 然后把所有hotel_mask 替换成 hotel_mask_sing_asp

4. **summ_dataset_factory.py里面添加一个条目**

5. **project_settings.py里面增加一个条目，并把里面的dataset_dir改掉**

6. 在checkpoints/下面的三个sum,clf,lm中各建立一个同名dataset文件夹

7. 在outputs/eval/下面建立一个文件夹

8. create  hotel_mask_sing_asp_dataset/processed/ 

9. Build subwordencoder. and put it into   hotel_mask_sing_asp_dataset/prceossed/subword_encoder/ 

10. **bash scripts/preprocess_data.sh hotel_mask_sing_asp_dataset.py**



6,7,9都成集成在了gene_sub_encoder&folders.sh里面