import pandas as pd
import os
import numpy as np


def get_user_item_inter():
    """处理ML100k数据集，筛选高质量交互、过滤低活跃度用户并离散化ID，保存为LightGCN可读格式。"""
    # 读取原始评分数据
    data = pd.read_csv('data/ML100k/ratings.csv')
    
    # 1. 筛选评分 >= 4 的记录（高质量交互）
    filtered_data = data[data['rating'] >= 4].copy()  # copy避免SettingWithCopyWarning
    
    # 2. 过滤交互数量少于3的用户（确保用户有足够历史行为）
    # 统计每个用户的交互次数
    user_inter_counts = filtered_data['userId'].value_counts()
    # 筛选出交互次数 >=3 的用户ID
    valid_users = user_inter_counts[user_inter_counts >= 3].index
    # 保留这些用户的交互数据
    filtered_data = filtered_data[filtered_data['userId'].isin(valid_users)]
    
    # 3. 离散化用户ID：将原始userId映射为连续整数（从0开始）
    unique_users = filtered_data['userId'].unique()
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    filtered_data['user_id'] = filtered_data['userId'].map(user_id_map)
    
    # 4. 离散化物品ID：同理映射为连续整数（从0开始）
    unique_items = filtered_data['movieId'].unique()
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    filtered_data['item_id'] = filtered_data['movieId'].map(item_id_map)
    
    # 5. 保留需要的列
    processed_data = filtered_data[['user_id', 'item_id', 'rating']]
    
    # 6. 保存处理后的数据
    processed_data.to_csv('data/ML100k/processed_interactions.csv', index=False)
    
    # 打印处理信息（增加过滤用户的统计）
    print(f"过滤后的数据量：{len(processed_data)} 条")
    print(f"过滤后用户数量：{len(unique_users)}（移除了交互次数<4的用户）")
    print(f"物品数量：{len(unique_items)}")
    print("处理完成，已保存至 processed_interactions.csv")

def get_train_and_test(processed_data, train_ratio=0.9, random_seed=42):
    """
    随机划分训练集和测试集（替代时间划分），确保每个用户的交互被随机分配
    """
    # 设置随机种子，保证划分可复现
    np.random.seed(random_seed)
    
    # 无需按时间排序，直接使用原始数据（但需按用户分组）
    data = processed_data.copy()
    
    train_list = []
    test_list = []
    
    for user_id, group in data.groupby('user_id'):
        n = len(group)
        # 确保每个用户至少有1条测试数据
        test_size = max(1, n - int(n * train_ratio))  # 测试集数量 = 总样本 - 训练集数量（训练集按比例取）
        
        # 随机抽取测试集的索引（无放回抽样）
        # 生成0到n-1的索引，随机打乱后取前test_size个作为测试集
        indices = np.arange(n)
        np.random.shuffle(indices)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        # 根据随机索引分割训练集和测试集
        train = group.iloc[train_indices][['user_id', 'item_id']]
        test = group.iloc[test_indices][['user_id', 'item_id']]
        
        train_list.append(train)
        test_list.append(test)
    
    # 合并为DataFrame
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    # # 验证：检查“必看物品”是否在测试集中出现
    # total_users = data['user_id'].nunique()
    # item_user_count = data.groupby('item_id')['user_id'].nunique()
    # mandatory_items = set(item_user_count[item_user_count >= total_users * 0.1].index)
    # test_mandatory = set(test_df['item_id'].unique()) & mandatory_items
    # print(f"必看物品: {mandatory_items}")
    # print(f"测试集中包含的必看物品: {test_mandatory}")  # 随机划分下，应大部分包含
    
    # 保存文件
    output_dir = 'data/real/ML100k'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_df.to_csv(f'{output_dir}/train.txt', sep=' ', header=False, index=False)
    test_df.to_csv(f'{output_dir}/test.txt', sep=' ', header=False, index=False)
    
    print(f"\n训练集规模：{len(train_df)} 条交互")
    print(f"测试集规模：{len(test_df)} 条交互")

def check():
    """检查处理后的数据集文件是否合法，有无数据泄露。"""
    # 1. 定义文件路径
    train_path = 'data/ML100k/real/train.txt'
    test_path = 'data/ML100k/real/test.txt'
    
    # 2. 检查文件是否存在
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: 训练集或测试集文件不存在！")
        return
    
    # 3. 读取训练集和测试集（格式：user_id item_id）
    try:
        train = pd.read_csv(train_path, sep=' ', header=None, names=['user_id', 'item_id'])
        test = pd.read_csv(test_path, sep=' ', header=None, names=['user_id', 'item_id'])
    except Exception as e:
        print(f"Error: 读取文件失败 - {e}")
        return
    
    # 4. 检查是否存在重复交互（同一用户-物品对同时出现在训练集和测试集）
    # 将交互转为元组集合，便于比较
    train_pairs = set(tuple(row) for row in train[['user_id', 'item_id']].values)
    test_pairs = set(tuple(row) for row in test[['user_id', 'item_id']].values)
    overlap = train_pairs & test_pairs  # 交集即为重复交互
    
    if overlap:
        print(f"Warning: 训练集与测试集存在 {len(overlap)} 条重复交互（数据泄露）！")
        print("部分重复交互示例：", list(overlap)[:5])  # 打印前5条
    else:
        print("✅ 训练集与测试集无重复交互，数据独立性验证通过。")
    
    # 5. 检查用户ID连续性（LightGCN通常要求用户/物品ID从0开始连续）
    all_users = set(train['user_id'].unique()) | set(test['user_id'].unique())
    max_user = max(all_users) if all_users else 0
    if set(all_users) != set(range(max_user + 1)):
        print(f"Warning: 用户ID不连续！存在空缺ID（最大用户ID为{max_user}，实际用户数为{len(all_users)}）。")
    else:
        print("✅ 用户ID连续（从0开始），格式验证通过。")
    
    # 6. 检查物品ID连续性
    all_items = set(train['item_id'].unique()) | set(test['item_id'].unique())
    max_item = max(all_items) if all_items else 0
    if set(all_items) != set(range(max_item + 1)):
        print(f"Warning: 物品ID不连续！存在空缺ID（最大物品ID为{max_item}，实际物品数为{len(all_items)}）。")
    else:
        print("✅ 物品ID连续（从0开始），格式验证通过。")
    
    # 7. 检查测试集用户是否均在训练集中出现（避免测试集有“新用户”，导致模型无法推荐）
    test_users = set(test['user_id'].unique())
    train_users = set(train['user_id'].unique())
    new_users_in_test = test_users - train_users
    if new_users_in_test:
        print(f"Warning: 测试集存在 {len(new_users_in_test)} 个训练集未出现的新用户，可能影响评估！")
        print("新用户ID示例：", list(new_users_in_test)[:5])
    else:
        print("✅ 测试集所有用户均在训练集中出现，符合推荐场景逻辑。")
    
    # 8. 检查是否存在空值
    if train.isnull().any().any() or test.isnull().any().any():
        print("Warning: 训练集或测试集中存在空值！")
    else:
        print("✅ 训练集和测试集无空值，数据完整性验证通过。")
    print(f"✅ 用户数量：{max_user+1}")
    print(f"✅ 物品数量：{max_item+1}")
    print("\n===== 检查完成 =====")

def generate_bandwagon_attack(
    target_item_id,
    ratio=0.01,
    begin_user_id=608,  # 真实用户最大ID（假用户从该值+1开始）
    top_k_num=20,
    top_k_list=[],  # 预计算的TopK热门物品列表
    real_interactions=None,  # 原始交互数据（DataFrame：含'user_id','item_id'列）
    save_path=None,  # 假样本保存路径（如"fake_data.txt"）
    interaction_per_fake_user=10
):
    """
    生成从众攻击假样本并保存为txt文件（格式：用户ID 物品ID）
    
    参数：
        target_item_id: 攻击目标物品ID
        ratio: 假样本占原始数据的比例
        begin_user_id: 真实用户最大ID（假用户ID起始值）
        top_k_num: 热门物品数量
        top_k_list: 预定义热门物品列表
        real_interactions: 原始交互数据（DataFrame，含'user_id','item_id'列）
        save_path: 保存路径（若为None则不保存）
    
    返回：
        fake_data: 假样本DataFrame（'user','item'）
    """
    # 校验输入
    if real_interactions is None:
        print("未提供原始交互数据real_interactions")
        return
    required_columns = ['user_id', 'item_id']
    if not all(col in real_interactions.columns for col in required_columns):
        print(f"原始数据必须包含列：{required_columns}")
        return
    if len(top_k_list) == 0:
        print("未提供热门物品列表，请检查输入")
        return
    
    # 处理热门物品列表
    top_k_list = top_k_list[:top_k_num]
    print(f"使用预定义Top{top_k_num}热门物品：{top_k_list[:5]}...")
    
    # 计算假样本总量（基于原始交互数据的行数）
    total_real = len(real_interactions['user_id'].unique())
    total_fake_user = max(1, int(total_real * ratio)) # 至少生成1条
    total_fake = total_fake_user * interaction_per_fake_user  # 每个假用户生成interaction_per_fake_user条交互
    print(f"原始用户数量：{total_real}，生成假样本用户数量：{total_fake_user}（占比{ratio*100}%）")
    
    # 假样本生成参数
    current_fake_user = begin_user_id + 1  # 假用户起始ID
    avg_interactions = 10  # 每个假用户平均交互数
    target_required = 1    # 必与目标物品交互1次
    remaining_per_user = avg_interactions - target_required
    
    fake_users = []
    fake_items = []
    
    # 生成假样本
    while len(fake_items) < total_fake:
        # 添加目标物品交互
        fake_users.append(current_fake_user)
        fake_items.append(target_item_id)
        current_count = 1
        
        # 填充剩余交互（优先热门物品）
        while current_count < avg_interactions and len(fake_items) < total_fake:
            # 80%概率选热门物品，20%选非热门
            if np.random.random() < 0.8:
                # 排除当前用户已交互的物品
                user_interacted = [target_item_id] + fake_items[-current_count:]
                candidate = [item for item in top_k_list if item not in user_interacted]
                if not candidate:
                    # 热门物品已用尽，随机选其他物品
                    all_items = set(real_interactions['item_id'].unique())  # 修正：使用'item_id'列
                    candidate = list(all_items - set(user_interacted))
            else:
                # 非热门物品（排除热门和已交互）
                all_items = set(real_interactions['item_id'].unique())  # 修正：使用'item_id'列
                non_top = all_items - set(top_k_list) - {target_item_id}
                user_interacted = fake_items[-current_count:]
                candidate = list(non_top - set(user_interacted))
            
            # 选一个物品添加
            if candidate:
                item = np.random.choice(candidate)
                fake_users.append(current_fake_user)
                fake_items.append(item)
                current_count += 1
            else:
                break  # 无候选物品时跳过
        
        current_fake_user += 1  # 下一个假用户
    
    # 去重（避免同一用户重复交互同一物品）
    fake_data = pd.DataFrame({'user': fake_users, 'item': fake_items})
    fake_data = fake_data.drop_duplicates(subset=['user', 'item'], keep='first')
    print(f"最终假样本量：{len(fake_data)}，假用户数量：{current_fake_user - begin_user_id - 1}")
    
    # 保存为txt文件（格式：用户ID 物品ID，空格分隔）
    if save_path:
        fake_data.to_csv(
            save_path,
            sep=' ',          # 空格分隔
            header=False,     # 无表头
            index=False,      # 无索引
            columns=['user', 'item']  # 确保列顺序
        )
        print(f"假样本已保存至：{save_path}")
    
    return fake_data


def generate_random_attack(
    target_item_id,
    ratio=0.01,
    begin_user_id=608,  # 真实用户最大ID（假用户从该值+1开始）
    real_interactions=None,  # 原始交互数据（DataFrame：含'user_id','item_id'列）
    save_path=None,  # 假样本保存路径（如"fake_data.txt"）
    p=0.7,  # 每次选择其他物品交互的概率（1-p为终止概率）
    max_interactions=20  # 单个假用户最大交互次数（避免无限循环）
):
    """
    生成随机攻击假样本（每个假用户必含目标物品，其余交互按概率p选择物品）
    
    参数：
        target_item_id: 攻击目标物品ID
        ratio: 假用户数量占原始用户数量的比例
        begin_user_id: 真实用户最大ID（假用户ID起始值）
        real_interactions: 原始交互数据（含'user_id','item_id'列）
        save_path: 假样本保存路径（None则不保存）
        p: 每次选择其他物品交互的概率（0 < p < 1）
        max_interactions: 单个假用户最大交互次数（防止无限循环）
    
    返回：
        fake_data: 假样本DataFrame（'user','item'）
    """
    # 输入校验
    if real_interactions is None:
        print("未提供原始交互数据real_interactions")
        return
    required_columns = ['user_id', 'item_id']
    if not all(col in real_interactions.columns for col in required_columns):
        print(f"原始数据必须包含列：{required_columns}")
        return
    if not (0 < p < 1):
        print("概率p必须满足0 < p < 1")
        return
    if max_interactions < 1:
        print("max_interactions必须大于等于1")
        return
    
    # 获取原始物品池（排除目标物品，用于生成其他交互）
    all_items = set(real_interactions['item_id'].unique())
    if target_item_id in all_items:
        other_items = list(all_items - {target_item_id})
    else:
        print(f"目标物品{target_item_id}不在原始物品池，已添加至假样本")
        other_items = list(all_items)  # 目标物品单独处理，不影响其他物品选择
    
    # 计算假用户数量
    total_real_users = len(real_interactions['user_id'].unique())
    total_fake_users = max(1, int(total_real_users * ratio))
    print(f"原始用户数量：{total_real_users}，生成假用户数量：{total_fake_users}（占比{ratio*100}%）")
    
    fake_users = []
    fake_items = []
    current_fake_user = begin_user_id + 1  # 假用户起始ID
    
    # 生成假样本
    for _ in range(total_fake_users):
        # 1. 每个假用户必含目标物品交互
        user_interactions = [target_item_id]
        fake_users.append(current_fake_user)
        fake_items.append(target_item_id)
        current_count = 1  # 已生成1条交互（目标物品）
        
        # 2. 按概率p生成其余交互，直到触发终止条件或达最大次数
        while current_count < max_interactions:
            # 以概率p继续选择物品，1-p终止
            if np.random.random() < p:
                # 选择未交互过的其他物品
                candidate = [item for item in other_items if item not in user_interactions]
                if not candidate:
                    break  # 无可用物品，终止当前用户生成
                
                # 随机选择一个候选物品
                selected_item = np.random.choice(candidate)
                user_interactions.append(selected_item)
                fake_users.append(current_fake_user)
                fake_items.append(selected_item)
            current_count += 1
        current_fake_user += 1  # 处理下一个假用户
    
    # 去重（避免同一用户重复交互同一物品）
    fake_data = pd.DataFrame({'user': fake_users, 'item': fake_items})
    fake_data = fake_data.drop_duplicates(subset=['user', 'item'], keep='first')
    print(f"最终假样本量：{len(fake_data)}，假用户数量：{current_fake_user - begin_user_id - 1}")
    
    # 保存假样本
    if save_path:
        fake_data.to_csv(
            save_path,
            sep=' ',
            header=False,
            index=False,
            columns=['user', 'item']
        )
        print(f"假样本已保存至：{save_path}")
    
    return fake_data

if __name__ == "__main__":
    # get_user_item_inter()
    # df = pd.read_csv('data/ML100k/processed_interactions.csv')
    # get_train_and_test(df,0.8)
    # check()
    top_items_ids = [402, 28, 16, 339, 13, 139, 22, 6, 20, 163, 4, 58, 479, 21, 484, 492, 373, 155, 354, 204]
    real_data_path = "data/ML100k/processed_interactions.csv"  
    real_interactions = pd.read_csv(
        real_data_path,
        usecols=['user_id', 'item_id']  # 只保留需要的列
    )
    # generate_bandwagon_attack(227,0.05,607,20,top_items_ids,real_interactions,"data/ML100k/fake_data.txt")
    generate_random_attack(227,0.05,607,real_interactions,"data/ML100k/fake_data.txt",0.99,10)
    pass