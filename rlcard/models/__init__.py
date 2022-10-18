''' Register rule-based models or pre-trianed models
'''
from rlcard.models.registration import load, register

register(
    model_id = 'uno-rule-v1',
    entry_point='rlcard.models.uno_rule_models:UNORuleModelV1')