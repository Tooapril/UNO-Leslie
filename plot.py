
import argparse
from rlcard.utils import plot_curve, plot_double_curve, plot_triple_curve

def plot(args):
    # Plot the learning curve
    plot_curve(args.csv1, args.fig, args.remark1)
    
    # Plot the double curve
    # csv0_path = 'experiments/uno/result/2/2_1/performance.csv'
    # csv1_path = 'experiments/uno/result/2/2_2/performance.csv'
    # csv2_path = 'experiments/uno/result/2/2_3/performance.csv'
    # fig_path = 'experiments/uno/result/2/fig3.png'
    
    # plot_double_curve(args.csv1, args.csv2, args.fig, args.remark1, args.remark2)
    plot_triple_curve(args.csv1, args.csv2, args.csv3, args.fig, args.remark1, args.remark2, args.remark3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plot the fig from the csv file")
    parser.add_argument('--csv1', type=str, default='experiments/uno/dqn/')
    parser.add_argument('--csv2', type=str, default='experiments/uno/dqn/')
    parser.add_argument('--csv3', type=str, default='experiments/uno/dqn/')
    parser.add_argument('--remark1', type=str, default='dmc_rule')
    parser.add_argument('--remark2', type=str, default='dmc_self')
    parser.add_argument('--remark3', type=str, default='dmc_self')
    parser.add_argument('--fig', type=str, default='experiments/uno/dqn/')
    
    args = parser.parse_args()
    plot(args)