import argparse
import configargparse
import os

def print_options(self, opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = self.parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.DefaultConfigFileParser
    description = 'project'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='deepmulticap')

    # general settings                              
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--name', type=str, default='deepmulticap', help='name of a model/experiment.')
    
    # Datasets related
    parser.add_argument('--log_path', type=str, default='./train_log')
    parser.add_argument('--dataroot', type=str, default='./data',
                        help='path to images (data folder)')
    parser.add_argument('--obj_path', type=str)
    parser.add_argument('--smpl_path', type=str)
    parser.add_argument('--tex_path', type=str)
    parser.add_argument('--loadSize', type=int, default=512, help='load size of input image')
    parser.add_argument('--b_min', nargs='+', default=[-3, -2, -3], type=float)
    parser.add_argument('--b_max', nargs='+', default=[3, 14, 3], type=float)
    parser.add_argument('--smpl_faces', type=str, default='/media/data1/shaoruizhi/Multiview_Pair/lib/data/smplx_fine.obj')

    # Experiment related
    parser.add_argument('--num_views', type=int, default=1, help='How many views to use for multiview network.')
    parser.add_argument('--random_multiview', action='store_true', help='Select random multiview combination.')
    parser.add_argument('--random_fix_view', action='store_true', help='Select random multiview combination.')

    # Training related
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')

    parser.add_argument('--num_threads', default=1, type=int, help='# sthreads for loading data')
    parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--pin_memory', action='store_true', help='pin_memory')

    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='adam learning rate')
    parser.add_argument('--num_epoch', type=int, default=1000, help='num epoch to train')

    
    parser.add_argument('--freq_plot', type=int, default=10, help='freqency of the save_checkpoints')
    parser.add_argument('--freq_save', type=int, default=50, help='freqency of the save_checkpoints')
    parser.add_argument('--freq_save_ply', type=int, default=100, help='freqency of the save ply')
    parser.add_argument('--freq_normal_show', type=int, default=1000, help='freqency of the save ply')

    # Testing related
    parser.add_argument('--resolution', type=int, default=256, help='# of grid in mesh reconstruction')
    parser.add_argument('--mc_threshold', type=float, default=0.5, help='marching cube threshold')

    # Sampling related
    parser.add_argument('--sigma', type=float, default=0.25, help='perturbation standard deviation for positions')
    parser.add_argument('--num_sample_inout', type=int, default=5000, help='# of sampling points')
    parser.add_argument('--visibility_sample', action='store_true')

    # Model related
    # General
    parser.add_argument('--norm', type=str, default='group',
                            help='instance normalization or batch normalization or group normalization')

    # hg filter specify
    parser.add_argument('--num_stack', type=int, default=4, help='# of hourglass')
    parser.add_argument('--num_hourglass', type=int, default=2, help='# of stacked layer of hourglass')
    parser.add_argument('--fine_num_stack', type=int, default=1, help='# of hourglass')
    parser.add_argument('--fine_num_hourglass', type=int, default=2, help='# of stacked layer of hourglass')
    parser.add_argument('--fine_hourglass_dim', type=int, default=32, help='feature dimension')
    parser.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
    parser.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
    parser.add_argument('--hourglass_dim', type=int, default='256', help='256 | 512')

    # Classification General
    parser.add_argument('--position_encoding', action='store_true', help='using position encoding')
    parser.add_argument('--mlp_dim', nargs='+', default=[323, 1024, 512, 256, 128, 1], type=int,
                            help='# of dimensions of mlp')
    parser.add_argument('--fine_mlp_dim', nargs='+', default=[355, 512, 256, 128, 1], type=int, help='# of dimensions of fine mlp')
    parser.add_argument('--use_tanh', action='store_true',
                            help='using tanh after last conv of image_filter network')

    # for train
    parser.add_argument('--random_flip', action='store_true', help='if random flip')
    parser.add_argument('--random_trans', action='store_true', help='if random flip')
    parser.add_argument('--random_scale', action='store_true', help='if random flip')
    parser.add_argument('--random_rotation', action='store_true', help='if random flip')
    parser.add_argument('--flip_normal', action='store_true', help='if smpl normal flip')
    parser.add_argument('--flip_smpl', action='store_true', help='if smpl 3d flip')
    parser.add_argument('--no_residual', action='store_true', help='no skip connection in mlp')
    
    parser.add_argument('--fine_part', action='store_true')
    parser.add_argument('--coarse_part', action='store_true')
    parser.add_argument('--mask_part', action='store_true')
    parser.add_argument('--preserve_single', action='store_true')

    # for infer
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--no_correct', action='store_true')
    parser.add_argument('--yaw_list', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--infer_reverse', action='store_true', help='infer rotation')
    parser.add_argument('--flip_x', action='store_true')

    # path
    parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='path to save checkpoints')
    parser.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to save checkpoints')
    parser.add_argument('--load_netN_checkpoint_path', type=str, default=None, help='path to save checkpoints')
    parser.add_argument('--load_optim_checkpoint_path', type=str, default=None, help='path to save checkpoints')
    parser.add_argument('--results_path', type=str, default='./results', help='path to save results ply')
    # for single image reconstruction
    parser.add_argument('--mask_path', type=str, help='path for input mask')
    parser.add_argument('--img_path', type=str, help='path for input image')

    # aug
    parser.add_argument('--aug_alstd', type=float, default=0.0, help='augmentation pca lighting alpha std')
    parser.add_argument('--aug_bri', type=float, default=0.0, help='augmentation brightness')
    parser.add_argument('--aug_con', type=float, default=0.0, help='augmentation contrast')
    parser.add_argument('--aug_sat', type=float, default=0.0, help='augmentation saturation')
    parser.add_argument('--aug_hue', type=float, default=0.0, help='augmentation hue')
    parser.add_argument('--aug_blur', type=float, default=0.0, help='augmentation blur')

    # debug
    parser.add_argument('--debug_3d', action='store_true')
    parser.add_argument('--debug_data', action='store_true')

    args, _ = parser.parse_known_args()

    return args