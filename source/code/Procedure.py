'''
创建于2020年3月1日
Xiangnan He等人提出的LightGCN的PyTorch实现
论文：LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
作者: Jianbai Ye (gusye@mail.ustc.edu.cn)

设计训练和测试流程
'''
import world  # 全局配置
import numpy as np  # 数值计算
import torch  # PyTorch框架
import utils  # 工具函数
import dataloader  # 数据加载器
from pprint import pprint  # 格式化打印
from utils import timer  # 计时工具
from time import time  # 时间模块
from tqdm import tqdm  # 进度条
import model  # 模型定义
import multiprocessing  # 多进程支持
from sklearn.metrics import roc_auc_score  # AUC计算


# 多进程测试时使用的CPU核心数（取总核心数的一半）
CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    """
    原始BPR训练流程（每次迭代生成正负样本对，计算BPR损失并更新模型）
    Args:
        dataset: 数据集实例
        recommend_model: 推荐模型（如LightGCN）
        loss_class: BPR损失管理器
        epoch: 当前训练轮次
        neg_k: 每个正样本对应的负样本数量（默认1）
        w: TensorBoard日志写入器（可选）
    Returns:
        训练信息字符串（包含损失和时间）
    """
    Recmodel = recommend_model
    Recmodel.train()  # 切换到训练模式（启用dropout等）
    bpr: utils.BPRLoss = loss_class  # BPR损失实例
    
    # 生成训练样本（用户-正样本-负样本三元组）
    with timer(name="Sample"):  # 计时：样本生成耗时
        S = utils.UniformSample_original(dataset)  # 调用采样函数生成三元组
    
    # 转换为PyTorch张量并移动到指定设备
    users = torch.Tensor(S[:, 0]).long()  # 用户ID
    posItems = torch.Tensor(S[:, 1]).long()  # 正样本物品ID
    negItems = torch.Tensor(S[:, 2]).long()  # 负样本物品ID

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    
    # 打乱样本顺序（增加随机性，提升训练效果）
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    
    # 计算总批次数
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.  # 累计损失，用于计算平均损失
    
    # 按批次训练
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        # 执行单步训练：计算损失、反向传播、参数更新
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri  # 累加损失
        
        # 若启用TensorBoard，记录每批的损失
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    
    # 计算平均损失
    aver_loss = aver_loss / total_batch
    # 获取各步骤的时间统计
    time_info = timer.dict()
    # 重置计时器
    timer.zero()
    # 返回训练信息（平均损失+时间）
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    """
    测试单个批次的用户推荐结果，计算评估指标
    Args:
        X: 元组 (排序后的物品列表, 真实交互物品列表)
    Returns:
        字典：包含recall、precision、ndcg在各top-k下的结果
    """
    sorted_items = X[0].numpy()  # 模型推荐的top-k物品（已排序）
    groundTrue = X[1]  # 该用户在测试集的真实交互物品
    
    # 将推荐结果转换为二值标签（1表示命中真实物品，0表示未命中）
    r = utils.getLabel(groundTrue, sorted_items)
    
    pre, recall, ndcg = [], [], []
    # 计算每个top-k对应的指标
    for k in world.topks:
        # 计算Recall@k和Precision@k
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        # 计算NDCG@k
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    
    return {
        'recall': np.array(recall), 
        'precision': np.array(pre), 
        'ndcg': np.array(ndcg)
    }
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    """
    测试模型性能，计算多个评估指标（Recall、Precision、NDCG等）
    Args:
        dataset: 数据集实例
        Recmodel: 推荐模型（如LightGCN）
        epoch: 当前轮次（用于日志记录）
        w: TensorBoard日志写入器（可选）
        multicore: 是否启用多进程测试（1启用，0禁用）
    Returns:
        字典：包含各指标在所有top-k下的平均值
    """
    u_batch_size = world.config['test_u_batch_size']  # 测试时的用户批大小
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict  # 测试集字典 {用户: 真实物品列表}
    Recmodel: model.LightGCN
    
    # 切换到评估模式（关闭dropout等）
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)  # 最大的top-k值（如50）
    
    # 若启用多进程，初始化进程池
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    
    # 初始化结果字典（存储各指标的累计值）
    results = {
        'precision': np.zeros(len(world.topks)),
        'recall': np.zeros(len(world.topks)),
        'ndcg': np.zeros(len(world.topks))
    }
    
    # 禁用梯度计算（测试阶段无需更新参数）
    with torch.no_grad():
        users = list(testDict.keys())  # 测试集中的所有用户
        try:
            # 检查批大小是否合理（避免批过大导致内存问题）
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"测试批大小过大，建议调整为 {len(users) // 10}")
        
        users_list = []  # 存储批次用户
        rating_list = []  # 存储批次推荐结果
        groundTrue_list = []  # 存储批次真实交互
        
        # 计算总批次数
        total_batch = len(users) // u_batch_size + 1
        
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
            
            # 释放内存（删除原始评分矩阵）
            del rating
            
            # 存储结果
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())  # 移动到CPU并存储
            groundTrue_list.append(groundTrue)
        
        # 验证批次数是否正确
        assert total_batch == len(users_list)
        
        # 打包推荐结果和真实标签，准备计算指标
        X = zip(rating_list, groundTrue_list)
        
        # 计算指标（单进程或多进程）
        pre_results = []
        if multicore == 1:
            # 多进程计算（加速大规模数据）
            pre_results = pool.map(test_one_batch, X)
        else:
            # 单进程计算
            for x in X:
                pre_results.append(test_one_batch(x))
        
        # 计算指标平均值（按用户数量加权）
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        
        # 除以总用户数，得到最终平均值
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        
        # 若启用TensorBoard，记录指标
        if world.tensorboard and w != None:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        
        # 关闭进程池（若启用多进程）
        if multicore == 1:
            pool.close()
        
        # 打印测试结果
        print(results)
        return results