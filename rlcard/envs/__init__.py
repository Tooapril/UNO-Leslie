''' Register new environments
'''
from rlcard.envs.env import Env
from rlcard.envs.registration import make, register

register(
    env_id='uno',
    entry_point='rlcard.envs.uno:UnoEnv',
)