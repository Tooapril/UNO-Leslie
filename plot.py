
import argparse
from rlcard.utils import plot_curve, plot_double_curve, plot_triple_curve

def plot(args):
    # Plot the learning curve
    # plot_curve(args.csvdir1, args.figdir, args.remark1)
    
    plot_double_curve(args.csvdir1, args.csvdir2, args.figdir, args.remark1, args.remark2)
    
    # plot_triple_curve(args.csvdir1, args.csvdir2, args.csvdir3, args.figdir, args.remark1, args.remark2, args.remark3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plot the fig from the csv file")
    parser.add_argument('--csvdir1', type=str, default='experiments/uno/dqn/')
    parser.add_argument('--csvdir2', type=str, default='experiments/uno/dqn/')
    parser.add_argument('--csvdir3', type=str, default='experiments/uno/dqn/')
    parser.add_argument('--figdir', type=str, default='experiments/uno/dqn/')
    parser.add_argument('--remark1', type=str, default='dqn')
    parser.add_argument('--remark2', type=str, default='dqn')
    parser.add_argument('--remark3', type=str, default='dqn')
    
    args = parser.parse_args()
    plot(args)