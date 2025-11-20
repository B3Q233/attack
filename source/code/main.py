import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join, exists

# ==============================
utils.set_seed(114514)
print(">>SEED:", world.seed)
# ==============================

import register
from register import dataset

# 初始化模型
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

# 定义损失函数
bpr = utils.BPRLoss(Recmodel, world.config)

# 模型保存路径设置
weight_file = utils.getFileName()  # 普通保存路径（可被覆盖）
best_weight_file = weight_file.replace(".pth", "_best.pth")  # 最佳模型路径
print(f"模型权重普通路径: {weight_file}")
print(f"最佳模型权重路径: {best_weight_file}")

# 模型加载逻辑（支持自定义权重路径）
load_path = ""
world.CUSTOM_WEIGHT_PATH  = None
world.CUSTOM_WEIGHT_PATH = "code\checkpoints\lgn-ML100k-3-150_best.pth.tar"
if hasattr(world, 'CUSTOM_WEIGHT_PATH') and world.CUSTOM_WEIGHT_PATH:
    # 优先使用自定义权重路径
    if exists(world.CUSTOM_WEIGHT_PATH):
        load_path = world.CUSTOM_WEIGHT_PATH
        cprint(f"检测到自定义权重路径: {load_path}")
    else:
        cprint(f"警告：自定义权重路径{world.CUSTOM_WEIGHT_PATH}不存在，将尝试其他加载方式")

if load_path is None and world.LOAD:
    # 未指定自定义路径或自定义路径无效时，使用默认路径
    if exists(weight_file):
        load_path = weight_file
    else:
        cprint(f"默认权重文件{weight_file}不存在，将从头开始训练")
# 执行模型加载
if load_path:
    try:
        # 支持跨设备加载权重
        state_dict = torch.load(load_path, map_location=world.device)
        model_state = Recmodel.state_dict()  # 获取新模型的状态字典
        matched_state = {}

        for name, param in state_dict.items():
            # 只处理名称匹配的参数
            if name in model_state:
                target_param = model_state[name]
                # 检查参数维度是否兼容（新增维度只能在第一维，且原始维度需匹配）
                if param.ndim > 0 and target_param.ndim > 0 and param.shape[1:] == target_param.shape[1:]:
                    # 原始参数长度（如用户嵌入的原始用户数）
                    src_len = param.shape[0]
                    tgt_len = target_param.shape[0]
                    if src_len <= tgt_len:
                        # 复制原始参数到新模型的对应位置
                        target_param[:src_len] = param
                        # 更新状态字典中该参数的值
                        matched_state[name] = target_param
                    else:
                        # 若原始参数长度大于新模型（理论上不应出现，因新模型已扩展）
                        cprint(f"参数{name}原始长度({src_len})大于新模型长度({tgt_len})，跳过加载")
                else:
                    cprint(f"参数{name}维度不匹配（原始：{param.shape}，新模型：{target_param.shape}），跳过加载")
        
        # 将匹配的参数加载到模型中
        Recmodel.load_state_dict(matched_state, strict=False)
        cprint(f"成功加载模型权重：{load_path}，共加载{len(matched_state)}个参数")

    except FileNotFoundError:
        cprint(f"模型文件{load_path}不存在，将从头开始训练")
    except Exception as e:
        cprint(f"加载模型失败: {str(e)}，将从头开始训练")
Neg_k = 10

# TensorBoard配置
if world.tensorboard:
    log_dir = join(world.BOARD_PATH,
                  time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    w = SummaryWriter(log_dir)
    cprint(f"已启用TensorBoard，日志路径: {log_dir}")
else:
    w = None
    world.cprint("未启用TensorBoard可视化")

# 跟踪最佳评估指标
best_hr = 0.0  # 初始最佳HR值
target_topk = world.topks[0]  # 目标K值（从配置中获取）
cprint(f"将跟踪最佳HR@{target_topk}作为模型保存指标")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        
        # 每10个epoch进行一次测试
        if epoch % 10 == 0 and epoch>0:
            cprint("[TEST]")
            metrics = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            
            # 提取当前HR指标（确保与Procedure.Test的返回格式匹配）
            current_hr = metrics['recall'].item()
            
            print(f"当前HR@{target_topk}: {current_hr:.4f}, 最佳HR@{target_topk}: {best_hr:.4f}")
            
            # 更新最佳模型
            if current_hr > best_hr:
                best_hr = current_hr
                torch.save(Recmodel.state_dict(), best_weight_file)
                cprint(f"HR@{target_topk}提升至{best_hr:.4f}，已保存最佳模型至{best_weight_file}")
        
        # 训练过程
        output_information = Procedure.BPR_train_original(
            dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w
        )
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        print(f"耗时: {time.time() - start:.2f}秒")
        
        # 保存当前轮次模型
        torch.save(Recmodel.state_dict(), weight_file)

finally:
    if world.tensorboard:
        w.close()
    # 训练结束总结
    cprint(f"训练结束，最佳HR@{target_topk}为: {best_hr:.4f}")
    cprint(f"最佳模型保存路径: {best_weight_file}")
    cprint(f"最后一轮模型保存路径: {weight_file}")