
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
        return winner
