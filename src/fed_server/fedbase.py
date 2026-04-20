import numpy as np
import torch
import time
from src.fed_client.client import BaseClient
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import copy
from src.utils.metrics import Metrics
from src.utils.tools import apply_feature_skew, build_client_feature_skews, get_runtime_device
from src.plugins import build_server_plugin
import torch.nn.functional as F
criterion = F.cross_entropy


class BaseFederated(object):
    def __init__(self, options, dataset, clients_label, model=None, optimizer=None,
                 model_builder=None, optimizer_builder=None, name=''):
        if model is not None and optimizer is not None:
            self.model = model
            self.optimizer = optimizer
        self.options = options
        self.dataset = dataset
        self.clients_label = clients_label
        self.model_builder = model_builder
        self.optimizer_builder = optimizer_builder
        if self.model_builder is None or self.optimizer_builder is None:
            raise ValueError('Both model_builder and optimizer_builder are required.')
        self.device = get_runtime_device(options)
        self.gpu = self.device.type != 'cpu'
        self.batch_size = options['batch_size']#batch_size=每轮训练用多少条样本
        self.num_round = options['round_num']#round_num=联邦学习总共训练多少轮
        self.per_round_c_fraction = options['c_fraction']#c_fraction=每轮训练用多少个客户端
        self.client_feature_skews = build_client_feature_skews(len(self.clients_label), options)#client_feature_skews=每个客户端的数据偏移量
        self.clients = self.setup_clients(self.dataset, self.clients_label)#根据数据集和每个客户端的数据索引真正把所有客户端创建出来
        self.clients_num = len(self.clients)#clients_num=客户端总数
        self.name = '_'.join([name, f'wn{int(self.per_round_c_fraction * self.clients_num)}',
                              f'tn{len(self.clients)}'])#name=实验名称
        self.metrics = Metrics(options, self.clients, self.name)#初始化实验指标记录器
        self.latest_global_model = copy.deepcopy(self.get_model_parameters())#latest_global_model=最新全局模型
        self.server_plugin = build_server_plugin(options, self.device)#根据配置创建服务器端插件。


#我这个 FedAvgTrainer 已经知道要用什么模型、什么优化器、哪些数据、哪些客户端划分规则了。
#现在把这些信息交给 BaseFederated，让它帮我把整个联邦训练基础设施建起来，包括客户端、全局参数、测试模块和插件系统。
#model：当前已经创建好的那个服务器模型对象
#model_builder：以后还能继续新建模型的函数
#self.optimizer：当前绑定服务器模型参数的优化器对象
#optimizer_builder：以后还能给别的模型参数创建优化器的函数


    @staticmethod
    def move_model_to_gpu(model, options):
        device = get_runtime_device(options)
        model.to(device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            print('>>> Use gpu on device {}'.format(device))
        elif device.type == 'mps':
            print('>>> Use Apple GPU via MPS')
        else:
            if options['gpu'] is True:
                print('>>> GPU requested but no accelerator available, using CPU instead')
            else:
                print('>>> Don not use gpu')

    def get_model_parameters(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_parameters(self, model_parameters_dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_parameters_dict[key]
        self.model.load_state_dict(state_dict)

    def train(self):
        """The whole training procedure

        No returns. All results all be saved.
        """
        raise NotImplementedError

    def setup_clients(self, dataset, clients_label):
        train_data = dataset.train_data
        train_label = dataset.train_label#这两行是先把全局训练数据和全局训练标签取出来，方便后面切分。
        all_client = []#创建列表用来存放所有客户端对象
        for i in range(len(clients_label)): #遍历每个客户端
            local_indices = self.clients_label[i]#获取当前客户端对应的数据索引列表
            local_train_data = apply_feature_skew(train_data[local_indices], self.client_feature_skews[i])
        #从全局训练集里取出属于这个客户端的数据，对这个客户端的数据施加它专属的特征偏移
            local_train_label = train_label[local_indices]#取出这个客户端对应的标签。
            local_model = self.model_builder()#创建一个本地模型，这个模型是基于全局模型构建的，但是每个客户端的模型参数都是独立的。
            local_model.to(self.device)
            local_optimizer = self.optimizer_builder(local_model.parameters())#创建一个本地优化器，这个优化器是基于全局优化器构建的，但是每个客户端的优化器参数都是独立的。
    #优化器是“根据损失函数计算出来的梯度，去更新模型参数”的工具。
    #训练四步：1.模型前向计算，得到预测结果；2.根据预测和真实标签算出损失；3.反向传播得到每个参数怎么改；（优化器负责执行“改参数”这一步）
            local_dataset = TensorDataset(
                torch.tensor(local_train_data, dtype=torch.float32),
                torch.tensor(local_train_label, dtype=torch.long),
            )#把本地数据和本地标签打包成一个TensorDataset对象，这个对象可以被DataLoader直接使用。
            local_client = BaseClient(
                self.options,#全局配置
                i,#客户端ID
                local_dataset,#本地数据
                local_model,#本地模型
                local_optimizer,#本地优化器
            )
            all_client.append(local_client)#把当前客户端对象添加到列表中

        return all_client

    def local_train(self, round_i, select_clients, ):  #本轮被抽中的客户端对象列表
        local_model_paras_set = []  #每个客户端训练后上传的模型更新结果
        stats = []  #每个客户端训练后的统计信息，比如 loss、acc、time
        for i, client in enumerate(select_clients, start=1):
            client.set_model_parameters(self.latest_global_model)#把服务器当前保存的最新全局模型参数写入这个客户端的本地模型。
            client.set_learning_rate(self.optimizer.param_groups[0]['lr'])#把服务器当前记录的学习率发给客户端。
            if self.server_plugin is not None:
                if hasattr(self.server_plugin, 'set_round_index'):
                    self.server_plugin.set_round_index(round_i)
                client.set_plugin_payload(self.server_plugin.build_broadcast_payload())#服务器额外信息下发。
            update, stat = client.local_train()
            local_model_paras_set.append(update)#本轮所有参与客户端的更新结果：模型参数+样本数+额外信息
            stats.append(stat)#本轮所有参与客户端的训练统计
            if True:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s ".format(
                       round_i, client.id, i, len(select_clients),
                       stat['loss'], stat['acc'] * 100, stat['time'], ))
        return local_model_paras_set, stats

    def aggregate_parameters(self, local_model_paras_set):
        # Each element is update dict: {"weights", "num_samples", "aux"}
        averaged_paras = copy.deepcopy(self.model.state_dict())
        train_data_num = 0
        for var in averaged_paras:
            averaged_paras[var] = 0
        for update in local_model_paras_set:
            num_sample = update["num_samples"]
            local_model_paras = update["weights"]
            for var in averaged_paras:
                averaged_paras[var] += num_sample * local_model_paras[var]
            train_data_num += num_sample
        for var in averaged_paras:
            averaged_paras[var] /= train_data_num
        if self.server_plugin is not None:
            self.server_plugin.aggregate_client_payloads(local_model_paras_set)
        return averaged_paras



    def test_latest_model_on_testdata(self, round_i):
        # Collect stats from total test data
        begin_time = time.time()
        stats_from_test_data = self.global_test(use_test_data=True)
        end_time = time.time()

        if True:
            print('= Test = round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_test_data['acc'],
                   stats_from_test_data['loss'], end_time-begin_time))
            print('=' * 102 + "\n")

        self.metrics.update_test_stats(round_i, stats_from_test_data)

#把当前最新的全局模型拿出来，在全局测试集上跑一遍，计算平均准确率、平均损失，并返回统计结果。
    def global_test(self, use_test_data=True):
        assert self.latest_global_model is not None  #运行前检查
        self.set_model_parameters(self.latest_global_model) #把最新全局参数装载回服务器模型
        self.model.eval()
        test_data = self.dataset.test_data
        test_label = self.dataset.test_label
        testDataLoader = DataLoader(  #这一段是把测试集包装成 PyTorch 可迭代的数据加载器。
            TensorDataset(
                torch.tensor(test_data, dtype=torch.float32),
                torch.tensor(test_label, dtype=torch.long),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for X, y in testDataLoader:
                if self.gpu:
                    X = X.to(self.device)
                    y = y.to(self.device)
                pred = self.model(X)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)

                correct = predicted.eq(y).sum()
                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        stats = {'acc': test_acc / test_total,
                 'loss': test_loss / test_total,
                 'num_samples': test_total,}
        return stats
