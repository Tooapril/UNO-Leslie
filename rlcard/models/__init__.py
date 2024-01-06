''' Register rule-based models or pre-trianed models
'''
from rlcard.models.registration import load, register

register(
    model_id = 'uno-rule-v1',
    entry_point='rlcard.models.uno_rule_v1:UNORuleModelV1')

register(
    model_id = 'uno-rule-v2',
    entry_point='rlcard.models.uno_rule_v2:UNORuleModelV2')

register(
    model_id = 'uno-rule-v3',
    entry_point='rlcard.models.uno_rule_v3:UNORuleModelV3')