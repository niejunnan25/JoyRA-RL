# JoyRA (New)

请clone到自己的目录下，checkout自己的分支，通过MR申请合并代码

## Environment
请不要自行修改starVLA conda环境

> source /mnt/workspace/envs/conda3/bin/activate starVLA_1

请修改输出路径为自己的目录

```bash
# 输出路径
run_root_dir=./outputs
run_id=robocasa_reproduce
# loss 可视化
# 目前是offline运行，再通过外网服务器同步到wandb
export WANDB_MODE=offline
export WANDB_DIR=/mnt/workspace/users/yuanzhihao/
```

## RoboCasa

Post-training
```bash
bash examples/Robocasa_tabletop/train_files/run_robocasa.sh
```

Test
```bash
bash examples/Robocasa_tabletop/eval_files/batch_eval_args.sh
# 统计
python summarize_eval_logs.py /mnt/workspace/users/yuanzhihao/code/starVLA/outputs/robocasa_reproduce/checkpoints/steps_100000_pytorch_model.pt.log/eval_20260104_192230/
```

### Result
Path
`
/mnt/workspace/users/yuanzhihao/code/starVLA/outputs/robocasa_reproduce/checkpoints/steps_100000_pytorch_model.pt
`
#### Setting
10w step, 4*8, bs 8, 24 task * 300 demo, 100 rollout

> Parsed success rate: 24/24  avg=0.5046

```
#    success  eps      env
----------------------------------------------------------------------------------------------------
1    0.8200   100      gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env:
2    0.6600   100      gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env:
3    0.5900   100      gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env:
4    0.5900   100      gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env:
5    0.5900   100      gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env:
6    0.5900   100      gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env:
7    0.5900   100      gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env:
8    0.5700   100      gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env:
9    0.5700   100      gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env:
10   0.5700   100      gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env:
11   0.5500   100      gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env:
12   0.5400   100      gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env:
13   0.5200   100      gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env:
14   0.5100   100      gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env:
15   0.4900   100      gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env:
16   0.4700   100      gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env:
17   0.4600   100      gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env:
18   0.4500   100      gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env:
19   0.4300   100      gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env:
20   0.4300   100      gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env:
21   0.3700   100      gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env:
22   0.3300   100      gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env:
23   0.2500   100      gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env:
24   0.1700   100      gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env:
```


## 宿迁真机

Pretrain

```bash
bash examples/Suqian_agibot/train_files/run_real.sh
```

Post-train
```bash
bash examples/Suqian_agibot/train_files/run_post_train.sh
```

##新的模型名称

```bash
QwenPeiceiver
```

##添加文件（其他文件暂时没用到应该是）

```bash
starVLA/model/framework/QwenPerceiver.py
starVLA/model/modules/action_model/PercieverHead.py
starVLA/model/modules/action_model/time_aware_action_head.py
```