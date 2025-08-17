import argparse
from args.utils import str2bool


def parse_pugan_args():
    parser = argparse.ArgumentParser(description='Training Arguments')
    # training
    parser.add_argument('--seed', default=21, type=float, help='seed')
    parser.add_argument('--optim', default='adamw', type=str, help='optimizer, adam or sgd')
    parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
    parser.add_argument('--base_lr', default=5.0e-4, type=float, help='base learning rate')
    parser.add_argument('--warm_lr', default=1.0e-6, type=float, help='warm learning rate')
    parser.add_argument('--final_lr', default=1.0e-6, type=float, help='final learning rate')
    parser.add_argument('--warm_lr_epochs', default=10, type=int, help='warm learning rate epochs')
    parser.add_argument('--weight_decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--lr_decay_step', default=20, type=int, help='learning rate decay step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='gamma for scheduler_steplr')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='workers number')
    parser.add_argument('--print_rate', default=200, type=int, help='loss print frequency in each epoch')
    parser.add_argument('--save_rate', default=5, type=int, help='model save frequency')
    parser.add_argument('--lr_scheduler', default='cosine', type=str, help='learning rate scheduler')

    # dataset
    parser.add_argument('--dataset', default='pugan', type=str, help='pu1k or pugan')
    parser.add_argument('--h5_file_path', default="./data/PU-GAN/train/PUGAN_poisson_256_poisson_1024.h5", type=str, help='the path of train dataset')
    parser.add_argument('--num_points', default=256, type=int, help='the points number of each input patch')
    parser.add_argument('--skip_rate', default=1, type=int, help='used for dataset')
    parser.add_argument('--use_random_input', default=False, type=str2bool, help='whether use random sampling for input generation')
    parser.add_argument('--jitter_sigma', type=float, default=0.01, help="jitter augmentation")
    parser.add_argument('--jitter_max', type=float, default=0.03, help="jitter augmentation")
    parser.add_argument('--data_augmentation', default=True, type=str2bool, help='whether use data augmentation')
    
    # # point transformer encoder
    parser.add_argument('--k', default=16, type=int, help='neighbor number in encoder')
    parser.add_argument('--encoder_dim', default=64, type=int, help='input(output) feature dimension in each dense block')
    parser.add_argument('--out_dim', default=64, type=int, help='input(output) feature dimension in each dense block')
    parser.add_argument('--encoder_bn', default=False, type=str2bool, help='whether use batch normalization in encoder')
    parser.add_argument('--global_mlp', default=True, type=str2bool, help='whether use global_mlp in encoder')

    #  kernel points
    # kernel_radius = kernel_point_receptive_radius * 1.5
    # conv_radius  = kernel_point_receptive_radius * 4
    parser.add_argument('--conv_radius', default=0.8, type=float, help='radius of kernel point convolution')
    parser.add_argument('--neighbor_limits', default=30, type=int, help='maximum number of points')
    parser.add_argument('--kernel_radius', default=0.3, type=float, help='radius of kernel point sphere')
    parser.add_argument('--kernel_shells_receptive_radius', default=0.2, type=float, help='receptive field of kernel shells convolution')
    parser.add_argument('--shell_sizes', default=[1, 15], type=list, help='kernel shells representation in kernel point convolution')
    parser.add_argument('--influence_mode', default='linear', type=str, help='specify the influence mode of kernel shells convolution')
    parser.add_argument('--kp_aggregation', default='nearest', type=str, help='specify the aggregation mode of kernel points, nearest or sum')
    parser.add_argument('--groups', default=7, type=int, help='number of groups in attention (negative value for ch_per_grp).')
    parser.add_argument('--fixed_kernel_points', default='center', type=str, help='specify the kernel points mode, center, none or verticals')
    parser.add_argument('--feature_dim', default=64, type=int, help='input feature dimension of kernel shells representation & kernel query generation')
    parser.add_argument('--kernel_output_dim', default=64, type=int, help='feature dimension in kernel shells convolution')
    parser.add_argument('--upsampling_rate', default=4, type=int, help='upsampling rate')
    
    # cross-attention
    parser.add_argument('--head_num', default=4, type=int, help='head number of attention')
    parser.add_argument('--trans_num', default=3, type=int, help='number of attention blocks')
    parser.add_argument('--trans_dim', default=64, type=int, help='dim of attention')
    parser.add_argument('--is_attn_bn', default=False, type=str2bool, help='whether use batch normalization in attention')
    
    # ouput
    parser.add_argument('--out_path', default='./output/pugan', type=str, help='the checkpoint and log save path')
    
    # test
    parser.add_argument('--patch_rate', default=3, type=int, help='used for patch generation')
    parser.add_argument('--input_dir', default='./data/PU1K/test/input_2048/input_2048/', type=str, help='path to folder of input point clouds')
    parser.add_argument('--gt_dir', default='./data/PU1K/test/input_2048/gt_8192/', type=str, help='path to folder of gt point clouds')
    parser.add_argument('--save_dir', default='./result', type=str, help='save upsampled point cloud and results')
    parser.add_argument('--ckpt', default='./pretrain/pu1k_best.pth', type=str, help='checkpoints')


    args = parser.parse_args()

    return args
