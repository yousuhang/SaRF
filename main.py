import argparse
import os
from networks.trainer_resnet import ResnetTrainer


def get_source_classifier_options(parser):
    ## Experiment Specific
    parser.add_argument('--checkpoints_folder', default='/storage/homefs/sy19t816/resnet_classification/scalp01mod', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--resume_iter', default=74)
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--n_epochs', default=201, type=int)

    ## Sizes
    parser.add_argument('--batch_size', default=16, type=int)
    # parser.add_argument('--ncolor_channels', default=1, type=int)
    parser.add_argument('--num_classes', default=6, type=int)

    ## Datasets

    parser.add_argument('--n_dataloader_workers', default=1)
    parser.add_argument('--train_data_path', default='/storage/homefs/sy19t816/sequence_scalped_h5/selected_Train_data.hd5f', type=str)
    parser.add_argument('--val_data_path', default='/storage/homefs/sy19t816/sequence_scalped_h5/selected_Valid_data.hd5f', type=str)
    parser.add_argument('--test_data_path',
                        default='/storage/homefs/sy19t816/sequence_scalped_h5/selected_Train_data.hd5f',
                        type=str)

    ## optimizer
    parser.add_argument("--lr", default=0.00001, type=float)

    ## display
    ## losses weight
    parser.add_argument("--alpha_1", default=0, type = float, help="weight on class saliency distinctiveness")
    parser.add_argument("--alpha_2", default=0, type = float, help="weight on pre_logits vs saliency per class similarity")
    parser.add_argument("--alpha_3", default=0, type=float, help="weight on saliency sparseness")
    ## test mode
    parser.add_argument("--test", default=False, type=bool, help="whether we enter in test mode")
    parser.add_argument("--vol_test", default=False, type=bool, help="whether we enter in test mode on volumes")
    parser.add_argument("--is_scalp", default=True, type=bool, help="whether the tested dataset is scalped")
    ## deep taylor
    parser.add_argument("--deep_taylor", default=False, type=bool, help="whether we calc deep taylor")
    parser.add_argument("--cal_all_sal", default=True, type=bool, help="whether to calc all saliency")
    parser.add_argument("--layer_ind", type = int, default = 13)
    parser.add_argument("--output_pre_logits", type = bool, default=False)
    opt = parser.parse_args()
    opt.gpu_id = 'cuda:%s' % opt.gpu_id
    if opt.alpha_1 < 1 and opt.alpha_2 < 1:
        opt.checkpoints_dir = f'{opt.checkpoints_folder}/scalp01mod_alpha1_{opt.alpha_1:.1f}_alpha2_{opt.alpha_2:.1f}_alpha3_{opt.alpha_3:.0f}'
    elif opt.alpha_1 < 1:
        opt.checkpoints_dir = f'{opt.checkpoints_folder}/scalp01mod_alpha1_{opt.alpha_1:.1f}_alpha2_{opt.alpha_2:.0f}_alpha3_{opt.alpha_3:.0f}'
    elif opt.alpha_2 < 1:
        opt.checkpoints_dir = f'{opt.checkpoints_folder}/scalp01mod_alpha1_{opt.alpha_1:.0f}_alpha2_{opt.alpha_2:.1f}_alpha3_{opt.alpha_3:.0f}'
    else:
        opt.checkpoints_dir = f'{opt.checkpoints_folder}/scalp01mod_alpha1_{opt.alpha_1:.0f}_alpha2_{opt.alpha_2:.0f}_alpha3_{opt.alpha_3:.0f}'
    opt.sal_dis_path = f'{opt.checkpoints_dir}/sal_dist'
    if opt.is_scalp:
        opt.scalped = '_scalped'
    else:
        opt.scalped = ''
    opt.deep_taylor_savepath = f'{opt.checkpoints_dir}/deep_taylor{opt.scalped}'
    opt.deep_taylor_savepath_layer = f'{opt.checkpoints_dir}/deep_taylor_layer{opt.scalped}/{opt.layer_ind}'
    if opt.deep_taylor:
        opt.test = True
    if opt.test:
        opt.continue_train = True
    if opt.output_pre_logits:
        opt.pre_logits_path = f'{opt.checkpoints_dir}/pred_logits{opt.scalped}'
    return opt

def ensure_dirs(opt):
    opt_dict = vars(opt)
    for key in opt_dict.keys():
        if 'folder' in key or 'dir' in key or 'path' in key:
            if not os.path.exists(opt_dict[key]):
                os.makedirs(opt_dict[key])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Classifier on Source Images')
    opt_style = get_source_classifier_options(parser)
    print(opt_style.checkpoints_dir)
    print(opt_style.resume_iter)
    ensure_dirs(opt_style)
    trainer = ResnetTrainer(opt_style)
    trainer.launch()
