
class UnoJudger:

    @staticmethod
    def judge_winner(payoffs):
        ''' Judge the winner of the game

        Args:
            players (list): The list of players who play the game

        Returns:
            (list): The player id of the winner
        '''
        _payoffs = [None for _ in range(2)] 
        _payoffs[0] = payoffs[0] + payoffs[2] # 计算队伍一总手牌值
        _payoffs[1] = payoffs[1] + payoffs[3] # 计算队伍二总手牌值
        
        if _payoffs[0] > _payoffs[1]: # 队伍一胜利
            return [0, 2]
        elif _payoffs[0] < _payoffs[1]: # 队伍一胜利
            return [1, 3]
        else: # 两队手牌值相同，都是 winner
            return None
