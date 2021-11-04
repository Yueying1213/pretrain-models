CUDA_VISIBLE_DEVICES=0 python3 -u continue_train.py \
                                --arch resnet \
                                --depth 18 \
                                --epoch 5 \
                                --batch-size 128 \
                                --lr 0.1 \
                                --lr-scheduler cosine \
			      --warmup \
                                --warmup-lr 0.0001 \
                                --warmup-epochs 10 \
                                --optmzr sgd \
                                --pretrain-model /home/yul21038/Reram/cifar-polar-share/model/cifar100_resnet18_acc_77.020_sgd.pt\
			#/data/imagenet
                             #   --rand-seed \
                                


