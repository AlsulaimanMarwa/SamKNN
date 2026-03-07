from __future__ import annotations

from river import metrics

import math


'''
The idea of this metric is fairness through unawareness -
the only attribute flipped in testing is the senstive one; all other attributes stay identical.
This means that ossible relationships between the sensitive attribute and other attributes are - deliberately - 
not part of the tested definition of fairness.
'''


class Fairness_Unawareness(metrics.base.Metric):

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

        self.num_samples = 0
        self.num_samples_diff = 0

        self.time_decay = time_decay

    def update(self, x, y_opp_pred, y_pred): #WARNING: y_opp_pred is the prediction of the "opposite" x

        y_pred = float(y_pred)
        y_opp_pred = float(y_opp_pred)

        #apply time_decay before each update
        self.num_samples *= (1.0-self.time_decay)
        self.num_samples_diff *= (1.0-self.time_decay)


        self.num_samples += 1

        if (y_opp_pred != y_pred):
            self.num_samples_diff += 1

        
    def get(self):
        try:
            #TODO: Work out a reasonable way to work that info into a score
            
            #for now, show pos sens div by pos control and same for neg

            
            return (self.num_samples_diff/self.num_samples)
        except ZeroDivisionError:
            return (math.inf)

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