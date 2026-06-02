# 一键化启动Robotwin测试

主要是两个脚本: `00_start_servers.sh`和`00_start_evals.sh`

在京东云上开一个8卡的notebook（CPU核和内存可以double，比如176和1920，这是optional的）来跑，目前不支持"训练任务"来测试。

0. 镜像选择`star_vla`，启动notebook，安装screen，直接`apt install screen`即可

1. 修改几处地方：
    
    1）修改`00_start_servers.sh`中的`JOYRA_ROOT`，`CKPT_TAG`和`MODEL_PATH`，都要对齐。

    2）修改`00_start_evals.sh`中的`JOYRA_ROOT`，`CKPT_TAG`和`MODEL_PATH`，都要对齐。

    3）给两个脚本都上权限：`chmod +x 00_start_servers.sh 00_start_evals.sh`

2. 先运行`00_start_servers.sh`（默认50个任务一次性并行，也可以调整脚本输入参数改成25，也就是2任务串行25个并行）。

    观察`nvidia-smi`或者`gpustat -i`，或者`screen -ls`进入对应的screen，来观察**权重是否都读上了**。（因为同时启动多个screen，load权重需要一定时间）当你观察到每张卡启动了6/7个任务后，就可以进入下一步。

3. 待确定权重都读上之后，运行`00_start_evals.sh`开始并行测试。会保存到对应的`eval-logs`里。

4. 在对应的log文件里看最后的结果，如果出错（比如websocket错误，虽然已经优化过）可以运行之后上传的debug版脚本，也可以直接运行子脚本`run_tasks_splitN.sh`传入参数`--rerun`即可，命令可以参考`00_start_evals.sh`。

5. 都测试完之后，建议直接关掉notebook，会重置环境。对于新测试，从step 0开始即可。

