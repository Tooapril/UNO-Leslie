from rlcard.utils import plot_curve, plot_double_curve


if __name__ == '__main__':
    # Plot the learning curve
    # csv_path = 'experiments/uno/dmc/position_0_2/performance.csv'
    # fig_path = 'experiments/uno/dmc/position_0_2/fig0.png'
    
    # plot_curve(csv_path, fig_path, 'dmc', 0)
    
    # Plot the double curve
    position = '0'
    csv0_path = 'experiments/uno/dmc/position_' + position + '_2/performance.csv'
    csv1_path = 'experiments/uno/dqn/performance_' + position + '.csv'
    fig_path = 'experiments/uno/result/fig' + position + '.png'
    
    plot_double_curve(csv0_path, csv1_path, fig_path, 'dmc', 'dqn', position)