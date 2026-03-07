from __future__ import annotations

from river import metrics

import math

class Demographic_Parity(metrics.base.Metric):

    _fmt = ",.6f"  # use commas to separate big numbers and show 6 decimals

    """
    protected_attribute must be tuple consisting of label and value in dict of protected attribute

    self.sensitive_feature is the feature of X that is sensitive, 
    self.sesitive_value is the value of this feature that denotes the protected class

    the _number attributes are there to count the (accepted) samples for sensitve and control group

    the time_decay is a factor that iteratively devalues previously seen samples from the accumulative impact

    """
    def __init__(self, protected_attribute, time_decay = 0):

        self.protected_attribute = protected_attribute
        self.sensitive_feature, self.sensitive_value = self.protected_attribute
        
        self.sensitive_accepted_number = 0
        self.sensitive_number = 0
        self.control_accepted_number = 0
        self.control_number = 0

        self.time_decay = time_decay

    def update(self, x, y_pred, y_true):

        #apply time_decay before each update
        self.sensitive_accepted_number *= (1.0-self.time_decay)
        self.sensitive_number *= (1.0-self.time_decay)
        self.control_accepted_number *= (1.0-self.time_decay)
        self.control_number *= (1.0-self.time_decay)


        #add new instance
        if x[self.sensitive_feature] == self.sensitive_value:
            self.sensitive_accepted_number += float(y_pred)
            self.sensitive_number += 1
        else:
            self.control_accepted_number += float(y_pred)
            self.control_number += 1
        
    def get(self):
        try:
            sensitive_acceptance_rate = self.sensitive_accepted_number/(self.sensitive_number)
            control_acceptance_rate = self.control_accepted_number/(self.control_number)
            epsilon = (control_acceptance_rate - sensitive_acceptance_rate)
            return epsilon
        except ZeroDivisionError:
            return math.inf
        
        
    def get_n_for_fabboo(self):
        try:
            control_acceptance_rate = self.control_accepted_number/self.control_number
            n = self.sensitive_number * control_acceptance_rate - self.sensitive_accepted_number
            n = int(n)
            return (n)
        except ZeroDivisionError:
            return 0
        
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