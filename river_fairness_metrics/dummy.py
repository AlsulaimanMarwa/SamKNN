from __future__ import annotations

from river import metrics

import math

class Dummy(metrics.base.Metric):

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
        self.time_decay = time_decay
        
        return

    def update(self, x, y_pred, y_true):

        return #do nothing
        
    def get(self):
        return 0 #returns default value 0
        
        
    def get_n_for_fabboo(self):
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