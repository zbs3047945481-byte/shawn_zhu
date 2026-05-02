#定义并解析联邦学习实验的所有超参数，把它们整理成一个 dict（options），供整个系统使用。
import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


#argparse 是 Python 的命令行参数解析库
#它允许你定义命令行参数，并自动解析这些参数，将它们转换为 Python 字典。
def input_options():
    parser = argparse.ArgumentParser()
    #•	创建一个“参数解析器”
	#•	后面所有 add_argument 都是在往这个解析器里注册参数
    # iid
    parser.add_argument('-is_iid', type=str2bool, default=False, help='data distribution is iid.')
    #是否使用 IID 数据分布
    
    parser.add_argument('--dataset_name', type=str, default='mnist', help='name of dataset.')
    #数据集名称 / 数据划分方式标识
    #mnist_dir_0.1 很可能表示：MNIST、Dirichlet α=0.1（Non-IID 程度）

    parser.add_argument('--model_name', type=str, default='mnist_cnn', help='the model to train')
    #指定用哪个模型
    #if model_name == 'mnist_cnn': model = MnistCNN()

    parser.add_argument('--gpu', type=str2bool, default=True, help='gpu id to use')
    #是否使用 GPU
    
    parser.add_argument('--round_num', type=int, default=301, help='number of round in comm')
    #通信轮数：每一轮 = 一次 FedAvg 聚合
    
    parser.add_argument('--num_of_clients', type=int, default=20, help='numer of the clients')
    #联邦系统中的客户端总数 K
    
    parser.add_argument('--c_fraction', type=float, default=0.2,
                        help='C fraction, 0 means 1 client, 1 means total clients')
    #每一轮参与训练的客户端比例
    
    parser.add_argument('--local_epoch', type=int, default=5, help='local train epoch')
    #每个客户端本地训练多少个 epoch
    #比如有1000条数据，把这1000个数据全部用来算一次梯度并更新参数，这完整一轮叫一个epoch

    parser.add_argument('--batch_size', type=int, default=256, help='local train batch size')
    #本地训练 batch size
    #每次梯度更新用多少条样本
    #比如有1000条数据，每轮训练只取100条数据来算梯度并更新参数，这100条数据叫一个batch
    parser.add_argument('--dataloader_num_workers', type=int, default=2,
                        help='Number of DataLoader worker processes.')
    parser.add_argument('--dataloader_pin_memory', type=str2bool, default=True,
                        help='Whether DataLoader should pin host memory when using GPU.')
    parser.add_argument('--torch_cudnn_benchmark', type=str2bool, default=True,
                        help='Enable cudnn benchmark for faster fixed-shape GPU training.')
    parser.add_argument('--early_stop_enable', type=str2bool, default=False,
                        help='Stop training when the global test accuracy has plateaued.')
    parser.add_argument('--early_stop_min_rounds', type=int, default=0,
                        help='Minimum communication rounds before early stopping is allowed.')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Number of evaluated rounds without meaningful improvement before stopping.')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='Minimum absolute accuracy improvement required to reset early-stop patience.')

    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, \
                        use value from origin paper as default")
    #本地学习率
    
    parser.add_argument('--gn0', type=int, default=1, help='gno')
    
    parser.add_argument('--seed', help='seed for randomness;', type=int, default=3001)
    #随机种子：控制数据划分、模型初始化，方便复现实验
    parser.add_argument('--experiment_tag', type=str, default='',
                        help='Optional tag appended to experiment output folder names.')
    
    parser.add_argument('--weight_decay', help='weight_decay;', type=int, default=1)
    #权重衰减：防止过拟合

    # ---------- Heterogeneity modeling ----------
    parser.add_argument('--partition_strategy', type=str, default='dirichlet',
                        choices=['iid', 'dirichlet'],
                        help='Client data partition strategy.')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3,
                        help='Dirichlet alpha for label skew. Smaller means stronger heterogeneity.')
    parser.add_argument('--unify_heterogeneity_alpha', type=str2bool, default=True,
                        help='Whether to let dirichlet_alpha jointly control label, quantity, and feature heterogeneity.')
    parser.add_argument('--min_samples_per_client', type=int, default=32,
                        help='Ensure each client owns at least this many training samples.')
    parser.add_argument('--enable_quantity_skew', type=str2bool, default=False,
                        help='Whether to vary client dataset sizes.')
    parser.add_argument('--quantity_skew_beta', type=float, default=0.5,
                        help='Dirichlet beta for client quantity skew. Smaller means more imbalance.')
    parser.add_argument('--enable_feature_skew', type=str2bool, default=False,
                        help='Whether to apply client-specific feature shift/noise.')
    parser.add_argument('--feature_alpha_anchor', type=float, default=0.1,
                        help='Reference alpha that corresponds to the strongest feature skew when alpha is unified.')
    parser.add_argument('--feature_max_scale_delta', type=float, default=0.15,
                        help='Maximum multiplicative scale deviation from 1.0 under the strongest unified feature skew.')
    parser.add_argument('--feature_max_bias_std', type=float, default=0.05,
                        help='Maximum additive bias std under the strongest unified feature skew.')
    parser.add_argument('--feature_max_noise_std', type=float, default=0.05,
                        help='Maximum additive Gaussian noise std under the strongest unified feature skew.')
    parser.add_argument('--feature_noise_std', type=float, default=0.05,
                        help='Std of client-specific additive Gaussian noise.')
    parser.add_argument('--feature_scale_low', type=float, default=0.85,
                        help='Lower bound of client-specific multiplicative scale.')
    parser.add_argument('--feature_scale_high', type=float, default=1.15,
                        help='Upper bound of client-specific multiplicative scale.')
    parser.add_argument('--feature_bias_std', type=float, default=0.05,
                        help='Std of client-specific additive bias.')

    # ---------- FedFed Feature Distillation Plugin (optional) ----------
    parser.add_argument('--use_fedfed_plugin', type=str2bool, default=False,
                        help='Enable FedFed-style feature distillation plugin.')
    parser.add_argument('--plugin_name', type=str, default='none',
                        choices=['none', 'fedfed_prototype', 'fedfed_image'],
                        help='Generic plugin selector. Prefer this over algorithm-specific toggles for new code.')
    parser.add_argument('--fedfed_input_channels', type=int, default=3,
                        help='Input channels used by image-space FedFed generator.')
    parser.add_argument('--fedfed_lambda_fd', type=float, default=0.3,
                        help='Weight of CE(f(x - q(x)), y) for image-space FedFed feature distillation.')
    parser.add_argument('--fedfed_lambda_norm', type=float, default=0.001,
                        help='Weight of the ||x - q(x)||^2 penalty for image-space FedFed.')
    parser.add_argument('--fedfed_lambda_shared', type=float, default=0.2,
                        help='Weight of CE on server-shared performance-sensitive features.')
    parser.add_argument('--fedfed_two_stage', type=str2bool, default=True,
                        help='Run paper-style FedFed: feature distillation first, then FedAvg over local plus shared features.')
    parser.add_argument('--fedfed_distill_rounds', type=int, default=30,
                        help='Communication rounds used for the feature distillation stage.')
    parser.add_argument('--fedfed_distill_local_epoch', type=int, default=2,
                        help='Local epochs used in each feature distillation round.')
    parser.add_argument('--fedfed_rho', type=float, default=0.3,
                        help='Relative norm budget rho for ||x_s|| <= rho * ||x|| in feature distillation.')
    parser.add_argument('--fedfed_lambda_rho', type=float, default=10.0,
                        help='Hinge penalty weight for violating the rho norm budget before clipping.')
    parser.add_argument('--fedfed_formal_online_distill', type=str2bool, default=False,
                        help='If false in two-stage mode, formal FedAvg only uses CE(x) and CE(shared_x_s).')
    parser.add_argument('--fedfed_hard_warmup_rounds', type=int, default=10,
                        help='Rounds used only for feature distillation before uploading/using shared x_s.')
    parser.add_argument('--fedfed_vae_latent_channels', type=int, default=64,
                        help='Latent channel width of the image-space beta-VAE generator.')
    parser.add_argument('--fedfed_lambda_recon', type=float, default=0.05,
                        help='Weight of reconstruction loss that keeps q(x) close to x.')
    parser.add_argument('--fedfed_beta_kl', type=float, default=0.001,
                        help='KL weight for the beta-VAE generator.')
    parser.add_argument('--fedfed_upload_per_class', type=int, default=20,
                        help='Max sensitive samples uploaded by one client for each class in one round.')
    parser.add_argument('--fedfed_upload_per_client', type=int, default=200,
                        help='Max sensitive samples uploaded by one client in one round.')
    parser.add_argument('--fedfed_shared_buffer_size', type=int, default=4000,
                        help='Max number of server-side shared sensitive samples.')
    parser.add_argument('--fedfed_shared_per_class_size', type=int, default=400,
                        help='Max server-side shared sensitive samples kept per class by FIFO.')
    parser.add_argument('--fedfed_shared_batch_size', type=int, default=256,
                        help='Batch size sampled from the shared sensitive feature buffer.')
    parser.add_argument('--fedfed_generator_weight_decay', type=float, default=0.0,
                        help='Weight decay for image-space FedFed generator optimizer.')
    parser.add_argument('--fedfed_sensitive_dim', type=int, default=64,
                        help='Dimension of performance-sensitive feature z_s (shared).')
    parser.add_argument('--fedfed_feature_dim', type=int, default=512,
                        help='Dimension of model intermediate feature h (e.g. mnist_cnn fc1 output).')
    parser.add_argument('--fedfed_clip_norm', type=float, default=1.0,
                        help='L2 clip norm for z_s before adding noise (privacy).')
    parser.add_argument('--fedfed_noise_sigma', type=float, default=0.1,
                        help='Gaussian noise std for z_s (privacy).')
    parser.add_argument('--fedfed_lambda_distill', type=float, default=1.0,
                        help='Weight of feature distillation loss L_distill.')
    parser.add_argument('--fedfed_distill_warmup_rounds', type=int, default=3,
                        help='Warm up prototype distillation over the first few communication rounds.')
    parser.add_argument('--fedfed_distill_count_tau', type=float, default=8.0,
                        help='Reliability temperature for local/global class counts in prototype distillation.')
    parser.add_argument('--fedfed_prototype_momentum', type=float, default=0.8,
                        help='EMA momentum for server-side global prototype updates. Higher is smoother.')
    parser.add_argument('--fedfed_use_cosine_distill', type=str2bool, default=True,
                        help='Whether to align prototypes with cosine distance instead of raw MSE.')
    parser.add_argument('--fedfed_normalize_prototypes', type=str2bool, default=True,
                        help='Whether to L2-normalize prototypes before sharing and distillation.')
    parser.add_argument('--fedfed_enable_projection', type=str2bool, default=True,
                        help='Whether to use the low-dimensional projection module before prototype sharing.')
    parser.add_argument('--fedfed_enable_prototype_sharing', type=str2bool, default=True,
                        help='Whether to upload, aggregate, and broadcast class prototypes across clients.')
    parser.add_argument('--fedfed_enable_distill', type=str2bool, default=True,
                        help='Whether to apply prototype distillation loss during local training.')
    parser.add_argument('--fedfed_enable_anchor', type=str2bool, default=True,
                        help='Whether to anchor local features to the round-start model features.')
    parser.add_argument('--fedfed_lambda_anchor', type=float, default=0.1,
                        help='Weight of the local feature anchor loss.')
    parser.add_argument('--fedfed_anchor_epoch_scaling', type=str2bool, default=False,
                        help='Scale anchor weight by local training intensity.')
    parser.add_argument('--fedfed_anchor_ref_epoch', type=float, default=5.0,
                        help='Local epoch value at which epoch-scaled anchor reaches fedfed_lambda_anchor_max.')
    parser.add_argument('--fedfed_enable_proto_cls', type=str2bool, default=False,
                        help='Apply cross-entropy classification loss on local backbone class prototypes.')
    parser.add_argument('--fedfed_lambda_proto_cls', type=float, default=0.1,
                        help='Weight of prototype classification loss.')
    parser.add_argument('--fedfed_enable_clip', type=str2bool, default=False,
                        help='Whether to clip prototype norm before upload.')
    parser.add_argument('--fedfed_enable_noise', type=str2bool, default=False,
                        help='Whether to add Gaussian noise to uploaded prototypes.')
    parser.add_argument('--fedfed_adaptive_control', type=str2bool, default=False,
                        help='Adapt distillation and anchor strengths from prototype quality and client drift signals.')
    parser.add_argument('--fedfed_lambda_distill_max', type=float, default=1.0,
                        help='Maximum distillation weight used by adaptive control.')
    parser.add_argument('--fedfed_lambda_anchor_max', type=float, default=0.1,
                        help='Maximum anchor weight used by adaptive control.')
    parser.add_argument('--fedfed_proto_stability_threshold', type=float, default=0.90,
                        help='Prototype cosine-stability threshold that marks server prototypes as reliable.')
    parser.add_argument('--fedfed_proto_coverage_threshold', type=float, default=0.80,
                        help='Reliable class coverage threshold that marks server prototypes as sufficiently complete.')
    parser.add_argument('--fedfed_adaptive_ramp_rounds', type=int, default=3,
                        help='Rounds used to ramp adaptive distillation after prototype quality becomes reliable.')
    parser.add_argument('--fedfed_anchor_drift_threshold', type=float, default=0.08,
                        help='Feature drift threshold above which adaptive anchor becomes active.')
    parser.add_argument('--fedfed_anchor_drift_slope', type=float, default=50.0,
                        help='Slope of the sigmoid gate used by adaptive anchor control.')
    parser.add_argument('--fedfed_num_classes', type=int, default=10,
                        help='Number of classes used to estimate prototype coverage.')
    
    args = parser.parse_args()
    #从命令行读取参数
    
    options = args.__dict__
    #把参数对象转成字典
    if options['is_iid']:
        options['partition_strategy'] = 'iid'
    if str(options['dataset_name']).lower() in {'cifar10', 'cifar-10'} and options['model_name'] == 'mnist_cnn':
        options['model_name'] = 'cifar_resnet18'
        options['fedfed_feature_dim'] = 512

    return options

#Batch size 决定“每次怎么学”，
#Epoch 决定“学多久”，
#Seed 决定“能不能复现”。
