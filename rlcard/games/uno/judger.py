
class UnoJudger:

    @staticmethod
    def judge_winner(payoffs):
        ''' Judge the winner of the game

        Args:
            players (list): The list of players who play the game

        Returns:
            (list): The player id of the winner
        '''
        winner = []
        _payoffs = [] # 计算队伍总手牌值
        _payoffs.append(payoffs[0] + payoffs[2])
        _payoffs.append(payoffs[1] + payoffs[3])
        
        for index, payoff in enumerate(payoffs):
            if len(set(_payoffs)) > 1: # 两队手牌值不同才有 winner
                if payoff == max(payoffs):
                    winner.append(index)
                    winner.append((index + 2) % 4) # 四人合作局，自己和队友同时胜利
        return winner
