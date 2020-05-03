1.注意生成 csv的时候 rating 需要是1-5的整数

2. 注意保证split已经可以泡通
3. 先lm，再sum，再test sum，再clf
4. 保存模型这些事情留到训练的时候进行
5. batchsize 小一些，保证不爆炸
6. 同时开5个ssh，都用nvtop看着
7. git 5个branch
8. 先训练aspect 3，再0，1，2，4；并且先等着1，训练完，训练完就训练下一个
9. 如果aspect 3 训练不完，至少aspect 0，1，2，4 肯定训练的完，大小只有原本的1/10-1/5 一个sum只需要4-8小时或者2-4小时



附录：batchsize

python3 pretrain_classifier.py --dataset=hotel_mask --model_type=cnn --clf_lr=0.0005 --cnn_n_feat_maps=256 --batch_size=128 --gpus=0,1,2,3



 python3 pretrain_lm.py --dataset='hotel_mask'  要在project settings里面改,zz用的48，cluster用48*4=192就可以了

![image-20200501102440899](/Users/saibo/Library/Application Support/typora-user-images/image-20200501102440899.png)

