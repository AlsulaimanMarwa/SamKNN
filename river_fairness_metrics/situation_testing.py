from __future__ import annotations

from river import metrics

import math


'''
The idea of this metric is situation testing/causal individual fairness in the simplified sense -
the only attribute flipped in testing is the senstive one; all other attributes stay identical.
This means that ossible relationships between the sensitive attribute and other attributes are - deliberately - 
not part of the tested definition of fairness.
'''


class Situation_Testing(metrics.base.Metric):

    _fmt = ",.6f"  # use commas to separate big numbers and show 6 decimals

    """
    protected_attribute must be tuple consisting of label and value in dict of protected attribute

    self.sensitive_feature is the feature of X that is sensitive, 
    self.sesitive_value is the value of this feature that denotes the protected class

    the _number attributes are there to count the TP, FP, positive (y_true) and negative (y_true) samples for sensitve and control group

    the time_decay is a factor that iteratively devalues previously seen samples from the accumulative impact

    """
    def __init__(self, protected_attribute, time_decay = 0.0):

        self.sensitive_feature, self.sensitive_value = protected_attribute

        self.real_sensitive_pos_number = 0
        self.real_sensitive_neg_number = 0

        self.flipped_sensitive_pos_number = 0
        self.flipped_sensitive_neg_number = 0

        self.real_control_pos_number = 0
        self.real_control_neg_number = 0
        
        self.flipped_control_pos_number = 0
        self.flipped_control_neg_number = 0

        self.time_decay = time_decay

    def update(self, x, y_true, y_pred): #WARNING: y_true is the prediction of the "opposite" x

        y_pred = float(y_pred)
        y_opp_pred = float(y_true)

        #apply time_decay before each update
        self.real_sensitive_pos_number *= (1.0-self.time_decay)
        self.real_sensitive_neg_number *= (1.0-self.time_decay)
        self.flipped_sensitive_pos_number *= (1.0-self.time_decay)
        self.flipped_sensitive_neg_number *= (1.0-self.time_decay)
        self.real_control_pos_number *= (1.0-self.time_decay)
        self.real_control_neg_number *= (1.0-self.time_decay)
        self.flipped_control_pos_number *= (1.0-self.time_decay)
        self.flipped_control_neg_number *= (1.0-self.time_decay)

        #add new instance
        if x[self.sensitive_feature] == self.sensitive_value:
            self.real_sensitive_pos_number += y_pred
            self.real_sensitive_neg_number += (1 - y_pred)

            self.flipped_control_pos_number += y_opp_pred
            self.flipped_control_neg_number += (1 - y_opp_pred)
        else:
            self.real_control_pos_number += y_pred
            self.real_control_neg_number += (1 - y_pred)

            self.flipped_sensitive_pos_number += y_opp_pred
            self.flipped_sensitive_neg_number += (1 - y_opp_pred)

        
    def get(self):
        try:
            #TODO: Work out a reasonable way to work that info into a score
            
            #for now, show pos sens div by pos control and same for neg

            pos_diff = (self.real_sensitive_pos_number + self.flipped_sensitive_pos_number)/(self.real_control_pos_number + self.flipped_control_pos_number)
            neg_diff = (self.real_sensitive_neg_number + self.flipped_sensitive_neg_number)/(self.real_control_neg_number + self.flipped_control_neg_number)

            return ('Positive Percentage: ' + str(pos_diff) + ', Negative Percentage: ' + str(neg_diff))
        except ZeroDivisionError:
            return ('TPR Difference: ' + str(math.inf) + 'FPR Difference: ' + str(math.inf))

    def revert(self):
        return RuntimeError
    
    def works_with(self, model):
        return True

    @property
    def bigger_is_better(self):
        return False
    
    @property
    def requires_labels(self):
        return True

    def revert(self):
        return RuntimeError
    
    def works_with(self, model):
        return True