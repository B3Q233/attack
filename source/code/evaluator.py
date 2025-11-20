import world  # 项目全局配置模块（包含超参数、设备设置等）
import utils  # 工具函数模块（包含数据处理、损失计算等）
from world import cprint  # 带颜色的打印函数
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
from tensorboardX import SummaryWriter  # 用于记录训练日志和可视化
import time  # 时间处理模块
import Procedure  # 训练/测试流程控制模块
from os.path import join  # 路径拼接工具
import register  # 模型和数据集注册器（统一管理模型与数据加载）
from register import dataset  # 加载注册的数据集
import model  # 模型定义模块（包含LightGCN等模型）

hot_top_dict = {}

user_size = 608

def get_result_evaluator( Recmodel, dataset,target):
    """
    自定义评估器加载特定参数文件后进行测试
    Args:
        Recmodel: 推荐模型实例
        dataset: 数据集实例
    Returns:
        None
    """
    u_batch_size = world.config['test_u_batch_size']  # 测试时的用户批大小
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict  # 测试集字典 {用户: 真实物品列表}
    Recmodel: model.LightGCN

    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    cnt = 0
    global hot_top_dict
    global user_size
    hot_top_dict = {}
    # 禁用梯度计算（测试阶段无需更新参数）
    with torch.no_grad():
        users = list(testDict.keys())  # 测试集中的所有用户
        try:
            # 检查批大小是否合理（避免批过大导致内存问题）
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"测试批大小过大，建议调整为 {len(users) // 10}")
        
        # 按批次处理测试用户
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            # 获取该批次用户的训练集正样本（用于过滤，避免推荐已交互物品）
            allPos = dataset.getUserPosItems(batch_users)
            # 获取该批次用户的测试集真实交互物品
            groundTrue = [testDict[u] for u in batch_users]
            
            # 将用户ID转换为张量并移动到设备
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            
            # 模型预测：获取用户对所有物品的评分
            rating = Recmodel.getUsersRating(batch_users_gpu)
            # 过滤掉用户已交互的物品（训练集中的正样本）
            exclude_index = []  # 要过滤的物品的行索引（用户索引）
            exclude_items = []  # 要过滤的物品ID
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))  # 重复用户索引（与物品数匹配）
                exclude_items.extend(items)  # 该用户已交互的物品
            # 将已交互物品的评分设为极小值（确保不会被推荐）
            rating[exclude_index, exclude_items] = -(1 << 10)
            
            # 获取评分最高的top-k物品（按max_K取，后续可兼容更小的k）
            _, rating_K = torch.topk(rating, k=max_K)
            for i in rating_K:
                for j in i:
                    if j.item() not in hot_top_dict:
                        hot_top_dict[j.item()]=1
                    else:
                        hot_top_dict[j.item()]+=1
                    if j.item() == target:
                        cnt += 1
            # 释放内存（删除原始评分矩阵）
            del rating
            
            # # 存储结果
            # users_list.append(batch_users)
            # rating_list.append(rating_K.cpu())  # 移动到CPU并存储
            # groundTrue_list.append(groundTrue)
        
        return cnt
    pass


if __name__ == "__main__":
    # 在模型初始化后，替换原有的加载逻辑
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)

    # 定义你要加载的特定参数文件路径
    specific_weight_path = "G:\\tj\\Paper\\lightGCN\\source\\code\\checkpoints\\lgn-ML100k-3-150_best.pth.tar"  # 替换为你的文件路径
    bpr = utils.BPRLoss(Recmodel, world.config)
    world.topks = [20]
    # 尝试加载特定参数
    try:
        # 加载参数文件（map_location确保设备兼容，如CPU/GPU）
        state_dict = torch.load(specific_weight_path, map_location=world.device)
        # 将参数加载到模型
        Recmodel.load_state_dict(state_dict)
        world.cprint(f"已成功加载特定参数: {specific_weight_path}")
        Procedure.Test(dataset, Recmodel, 0, None, 0)
        cnt = get_result_evaluator(Recmodel, dataset,227)
        sorted_hot_top = sorted(hot_top_dict.items(), key=lambda x: x[1], reverse=True)
        print(user_size)
        print(cnt/user_size)
        print(cnt)
        for i in range(20):
            print(f"物品ID: {sorted_hot_top[i][0]}, 出现次数: {sorted_hot_top[i][1]}")
    except FileNotFoundError:
        world.cprint(f"警告：特定参数文件 {specific_weight_path} 不存在，将使用随机初始化参数")