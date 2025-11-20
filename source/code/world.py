'''
创建于2020年3月1日
Xiangnan He等人提出的LightGCN的PyTorch实现
论文：LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

作者: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os  # 操作系统交互模块
from os.path import join  # 路径拼接函数
import torch  # PyTorch深度学习框架
from enum import Enum  # 枚举类支持
from parse import parse_args  # 命令行参数解析
import multiprocessing  # 多进程支持

# 解决KMP库重复加载的问题（Windows系统常见错误）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 解析命令行参数
args = parse_args()

# 定义项目核心路径
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))  # 项目根目录（上两级目录）
CODE_PATH = join(ROOT_PATH, 'code')  # 代码目录
DATA_PATH = join(ROOT_PATH, 'data')  # 数据目录
BOARD_PATH = join(CODE_PATH, 'runs')  # TensorBoard日志目录
FILE_PATH = join(CODE_PATH, 'checkpoints')  # 模型权重保存目录

# 将源代码目录添加到Python路径，确保模块可导入
import sys
sys.path.append(join(CODE_PATH, 'sources'))

# 创建模型权重保存目录（若不存在）
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)  # exist_ok=True避免目录已存在时报错


# 模型配置字典（存储超参数）
config = {}
# 支持的数据集列表
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book','ML100k']
# 支持的模型列表（mf:矩阵分解，lgn:LightGCN）
all_models  = ['mf', 'lgn']

# 从命令行参数加载配置（覆盖默认值）
config['bpr_batch_size'] = args.bpr_batch  # BPR损失的批处理大小
config['latent_dim_rec'] = args.recdim  # 嵌入维度
config['lightGCN_n_layers'] = args.layer  # LightGCN的层数
config['dropout'] = args.dropout  # 是否启用dropout
config['keep_prob'] = args.keepprob  # dropout的保留概率
config['A_n_fold'] = args.a_fold  # 邻接矩阵拆分的折叠数（用于大规模数据）
config['test_u_batch_size'] = args.testbatch  # 测试时用户的批处理大小
config['multicore'] = args.multicore  # 是否启用多进程测试
config['lr'] = args.lr  # 学习率
config['decay'] = args.decay  # 权重衰减系数（正则化）
config['pretrain'] = args.pretrain  # 是否使用预训练嵌入
config['A_split'] = False  # 是否拆分邻接矩阵（默认关闭）
config['bigdata'] = False  # 是否处理大规模数据（默认关闭）

# 检查GPU可用性并设置设备
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")  # 优先使用GPU，否则使用CPU

# 计算可用的CPU核心数（取总核心数的一半用于多进程）
CORES = multiprocessing.cpu_count() // 2
# 随机种子（确保实验可复现）
seed = args.seed

# 从命令行参数获取数据集和模型名称
dataset = args.dataset
model_name = args.model

# 验证数据集和模型是否支持
if dataset not in all_dataset:
    raise NotImplementedError(f"暂不支持数据集 {dataset}，请尝试以下数据集：{all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"暂不支持模型 {model_name}，请尝试以下模型：{all_models}")


# 训练相关配置
TRAIN_epochs = args.epochs  # 训练总轮次
LOAD = args.load  # 是否加载预训练模型
PATH = args.path  # 预训练模型路径
topks = eval(args.topks)  # 评估的Top-K值（如[10, 20, 50]）
tensorboard = args.tensorboard  # 是否启用TensorBoard可视化
comment = args.comment  # 实验注释（用于日志区分）

# 忽略pandas的FutureWarning警告
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)


def cprint(words: str):
    """带黄色背景的打印函数，用于突出显示重要信息"""
    print(f"\033[0;30;43m{words}\033[0m")  # ANSI转义码：黑色文字+黄色背景


# 程序启动logo（ASCII艺术字）
logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# 字体来源：ANSI Shadow
# 参考：http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# 打印logo
print(logo)