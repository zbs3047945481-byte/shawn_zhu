import random

import numpy as np
import torch


def get_runtime_device(options=None):
    options = options or {}
    use_accelerator = bool(options.get('gpu', False))
    if not use_accelerator:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def accelerator_available():
    if torch.cuda.is_available():
        return True
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return True
    return False


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(options=None):
    options = options or {}
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(options.get('torch_cudnn_benchmark', True))


def resolve_heterogeneity_options(options=None):
    options = dict(options or {})
    if not options.get('unify_heterogeneity_alpha', False):
        return options

    alpha = float(options.get('dirichlet_alpha', 0.3))
    alpha = max(alpha, 1e-6)

    # Use one alpha to control all heterogeneity sources:
    # smaller alpha => stronger label skew, stronger quantity skew, stronger feature shift.
    options['quantity_skew_beta'] = alpha

    feature_anchor = max(float(options.get('feature_alpha_anchor', 0.1)), 1e-6)
    feature_strength = min(1.0, feature_anchor / alpha)

    max_scale_delta = float(options.get('feature_max_scale_delta', 0.15))
    max_bias_std = float(options.get('feature_max_bias_std', 0.05))
    max_noise_std = float(options.get('feature_max_noise_std', 0.05))

    scale_delta = max_scale_delta * feature_strength
    options['feature_scale_low'] = 1.0 - scale_delta
    options['feature_scale_high'] = 1.0 + scale_delta
    options['feature_bias_std'] = max_bias_std * feature_strength
    options['feature_noise_std'] = max_noise_std * feature_strength
    options['feature_unified_strength'] = feature_strength
    return options


def _split_by_counts(indices, counts):
    assignments = []
    start = 0
    for count in counts:
        end = start + count
        assignments.append(indices[start:end].tolist())
        start = end
    return assignments


def _ensure_min_samples(assignments, min_samples):
    if min_samples <= 0:
        return assignments

    lengths = [len(items) for items in assignments]#每个客户端的数据量
    for client_id, current_len in enumerate(lengths):#遍历每个客户端，确保每个客户端的数据量至少为min_samples
        while current_len < min_samples:#如果当前客户端的数据量小于 min_samples，就继续移动数据
            donor_id = int(np.argmax(lengths))#找到数据量最大的客户端
            if donor_id == client_id or lengths[donor_id] <= min_samples:#如果 donor_id 就是当前客户端，或者 donor_id 的数据量已经小于等于 min_samples，就跳出循环
                break
            assignments[client_id].append(assignments[donor_id].pop())#把 donor_id 的数据移动到 client_id
            lengths[client_id] += 1#client_id 的数据量增加1
            lengths[donor_id] -= 1#donor_id 的数据量减少1
            current_len += 1#current_len 增加1
    return assignments


def _sample_client_capacities(client_num, total_num, beta, min_samples):
    if client_num * min_samples > total_num:
        raise ValueError('min_samples_per_client is too large for the current dataset size.')

    base = np.full(client_num, min_samples, dtype=int)
    remaining = total_num - base.sum()
    if remaining <= 0:
        return base

    weights = np.random.dirichlet(np.full(client_num, beta))
    extra = np.random.multinomial(remaining, weights)
    return base + extra


def _build_iid_partition(train_labels, client_num, min_samples, enable_quantity_skew, quantity_skew_beta):
    shuffled_indices = np.random.permutation(len(train_labels))
    if enable_quantity_skew:
        counts = _sample_client_capacities(client_num, len(train_labels), quantity_skew_beta, min_samples)
    else:
        counts = np.full(client_num, len(train_labels) // client_num, dtype=int)
        counts[:len(train_labels) % client_num] += 1
    return _split_by_counts(shuffled_indices, counts)

#模拟label skew和quantity skew两种数据异质性
def _build_dirichlet_partition_once(labels, client_num, alpha, enable_quantity_skew, quantity_skew_beta):
    assignments = [[] for _ in range(client_num)] #长度为client_num的列表，每个元素是一个空列表，用来存放每个客户端的数据索引
    client_activity = np.ones(client_num, dtype=float) #ones：创建一个全是1的数组，dtype=float：指定数组元素类型为浮点数
    if enable_quantity_skew: #是否允许不同客户端间样本数量不一致
        client_activity = np.random.dirichlet(np.full(client_num, quantity_skew_beta))#创建一个长度为 client_num、每个元素都等于quantity_skew_beta的数组。
#dirichlet两个约束：所有值不可为负；所有值加和为1。

    for cls in np.unique(labels): #每次处理一类，np.unique()：返回数组中所有不重复的值，并按升序排列。
        class_indices = np.where(labels == cls)[0] #np.where()：返回数组中满足条件的元素的索引。
        np.random.shuffle(class_indices) #原地打乱数组顺序
        class_weights = np.random.dirichlet(np.full(client_num, alpha)) #为“当前这个类别”随机生成一个客户端分配比例。label skew核心
        class_weights = class_weights * client_activity #label skew和quantity skew融合起来
        class_weights = class_weights / class_weights.sum() #上一步打乱了和为1，重新归一化，使其和为1
        split_points = (np.cumsum(class_weights) * len(class_indices)).astype(int)[:-1] #np.cumsum()：返回数组中每个元素的累积和。
#sequence[start:stop:step]  start：从哪里开始；stop：到哪里结束，不含stop本身；step：步长。-1表示最后一个元素
        #astype(int)：将结果转换为整数类型。[:-1]：去掉最后一个元素。 这一行的目的就是：把比例转换成切分位置
        splits = np.split(class_indices, split_points) #np.split()：将数组分割为多个子数组。
        for client_id, split in enumerate(splits): #enumerate()：返回索引和对应的值。
            assignments[client_id].extend(split.tolist()) #tolist()：将NumPy数组转换为Python列表。
    return assignments


def _build_dirichlet_partition(train_labels, client_num, alpha, min_samples, enable_quantity_skew, quantity_skew_beta):
    labels = np.asarray(train_labels) #np.asarray() 将输入转换为NumPy数组
    max_retries = 100
    for _ in range(max_retries):
        assignments = _build_dirichlet_partition_once(
            labels,
            client_num,
            alpha,
            enable_quantity_skew,
            quantity_skew_beta,
        )
        if min_samples <= 0 or min(len(items) for items in assignments) >= min_samples: #没最小标准或者所有项都达标
            return assignments

    raise ValueError(
        'Failed to build a Dirichlet partition satisfying min_samples_per_client={} '
        'after {} retries. Consider increasing dirichlet_alpha, reducing min_samples_per_client, '
        'or reducing num_of_clients.'.format(min_samples, max_retries)
    )


def get_each_client_data_index(train_labels, client_num, options=None):
    options = options or {}
    strategy = options.get('partition_strategy', 'dirichlet')
    min_samples = options.get('min_samples_per_client', 0)
    enable_quantity_skew = options.get('enable_quantity_skew', False) #是否启用数量偏斜
    quantity_skew_beta = options.get('quantity_skew_beta', 1.0) #数量偏斜参数

    if strategy == 'iid':
        return _build_iid_partition(
            train_labels,
            client_num,
            min_samples,
            enable_quantity_skew,
            quantity_skew_beta,
        )

    if strategy == 'dirichlet':
        return _build_dirichlet_partition(
            train_labels,
            client_num,
            options.get('dirichlet_alpha', 0.3),
            min_samples,
            enable_quantity_skew,
            quantity_skew_beta,
        )

    raise ValueError('Unsupported partition strategy: {}'.format(strategy))


def build_client_feature_skews(client_num, options):
    if not options.get('enable_feature_skew', False):
        return [None] * client_num

    low = options.get('feature_scale_low', 1.0)
    high = options.get('feature_scale_high', 1.0)
    bias_std = options.get('feature_bias_std', 0.0)
    noise_std = options.get('feature_noise_std', 0.0)

    skews = []
    for _ in range(client_num):
        skews.append({
            'scale': float(np.random.uniform(low, high)),
            'bias': float(np.random.normal(0.0, bias_std)),
            'noise_std': float(max(noise_std, 0.0)),
        })
    return skews


def apply_feature_skew(data, skew):
    if skew is None:
        return data.copy()

    transformed = data.astype(np.float32).copy()
    transformed = transformed * skew['scale'] + skew['bias']
    if skew['noise_std'] > 0:
        transformed = transformed + np.random.normal(
            0.0,
            skew['noise_std'],
            size=transformed.shape,
        ).astype(np.float32)
    return np.clip(transformed, 0.0, 1.0)
