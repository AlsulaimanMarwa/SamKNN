from river.metrics.base import Metric
import collections

class Metrics(Metric, collections.UserList):
    """A container class for handling multiple metrics at once.

    Parameters
    ----------
    metrics
    str_sep

    """

    def __init__(self, metrics, str_sep="\n"):
        super().__init__(metrics)
        self.str_sep = str_sep

    def update(self, x, y_true, y_pred, w=1.0) -> None:
        # If the metrics are classification metrics, then we have to handle the case where some
        # of the metrics require labels, whilst others need to be fed probabilities
        if hasattr(self, "requires_labels") and not self.requires_labels:
            for m in self:
                if m.requires_labels:
                    m.update(x=x, y_true=y_true, y_pred=max(y_pred, key=y_pred.get))
                else:
                    m.update(x=x, y_true=y_true, y_pred=y_pred)
            return

        for m in self:
            m.update(x=x, y_true=y_true, y_pred=y_pred)

    def revert(self, y_true, y_pred, w=1.0) -> None:
        # If the metrics are classification metrics, then we have to handle the case where some
        # of the metrics require labels, whilst others need to be fed probabilities
        if hasattr(self, "requires_labels") and not self.requires_labels:
            for m in self:
                if m.requires_labels:
                    m.revert(y_true, max(y_pred, key=y_pred.get), w)
                else:
                    m.revert(y_true, y_pred, w)
            return

        for m in self:
            m.revert( y_true, y_pred, w)

    def get(self):
        return [m.get() for m in self]

    def works_with(self, model) -> bool:
        return all(m.works_with(model) for m in self)

    @property
    def bigger_is_better(self):
        raise NotImplementedError

    @property
    def works_with_weights(self):
        return all(m.works_with_weights for m in self)

    @property
    def requires_labels(self):
        return all(m.requires_labels for m in self)

    def __repr__(self):
        return self.str_sep.join(str(m) for m in self)

    def __add__(self, other):
        try:
            other + self[0]  # Will raise a ValueError if incompatible
        except IndexError:
            pass
        self.append(other)
        return self

    def clone(self):
        return self.__class__([m.clone() for m in self])