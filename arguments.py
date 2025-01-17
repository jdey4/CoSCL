import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    # Arguments
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--experiment', default='pmnist', type=str, required=False,
                        choices=['split_cifar100_sc_5', 'split_cifar100_rs_5', 'split_cifar100_rs_2'],
                        help='(default=%(default)s)')
    parser.add_argument('--approach', default='lrp', type=str, required=False,
                        choices=['ewc', 'afec_ewc', 'ewc_coscl',
                                 'mas', 'afec_mas', 'mas_coscl',
                                 'si', 'rwalk',
                                 'gs', 'gs_coscl', 'hat', 'hat_coscl',
                                 'er', 'er_coscl',
                                 'ft', 'ft_coscl', 'jt', 'random_init'],
                        help='(default=%(default)s)')
    parser.add_argument('--mtl', default='coscl', type=str, required=False, choices=['coscl'], help='(default=%(default)s)')
    
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--nepochs', default=100, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--nepochs_pt', default=1, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch-size', default=256, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--rho', default=0.3, type=float, help='(default=%(default)f)')
    parser.add_argument('--gamma', default=0.75, type=float, help='(default=%(default)f)')
    parser.add_argument('--eta', default=0.8, type=float, help='(default=%(default)f)')
    parser.add_argument('--smax', default=400, type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb', default='1', type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb_emp', default='0', type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb1', default='0', type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb2', default='0', type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb3', default='0', type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb_full', default='0', type=float, help='(default=%(default)f)')
    parser.add_argument('--nu', default='0.1', type=float, help='(default=%(default)f)')
    parser.add_argument('--mu', default=0, type=float, help='groupsparse parameter')
    parser.add_argument('--s_gate', default='1', type=float, help='(default=%(default)f)')
    parser.add_argument('--use_TG', action='store_true', help='use task adaptive gate')


    parser.add_argument('--img', default=0, type=float, help='image id to visualize')

    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--tasknum', default=10, type=int, help='(default=%(default)s)')
    parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
    parser.add_argument('--sample', type = int, default=1, help='Using sigma max to support coefficient')

    parser.add_argument('--pdrop1',default=0.2,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--pdrop2',default=0.5,type=float,required=False,help='(default=%(default)f)')

    parser.add_argument('--warmup', default=60, type=int, required=False, help='warm up start epoch')
    parser.add_argument('--slot', default=0, type=int, required=True, help='slot number')
    parser.add_argument('--shift', default=1, type=int, required=True, help='shift number')

    args=parser.parse_args()
    return args

