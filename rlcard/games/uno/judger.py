
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
        for index, payoff in enumerate(payoffs):
            if payoff == max(payoffs):
                winner.append(index)
                winner.append((index + 2) % 4) # 四人合作局，自己和队友同时胜利
        return winner
