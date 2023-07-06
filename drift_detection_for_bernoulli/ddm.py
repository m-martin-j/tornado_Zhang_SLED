"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Drift Detection Method (DDM) Implementation ***
Paper: Gama, Joao, et al. "Learning with drift detection."
Published in: Brazilian Symposium on Artificial Intelligence. Springer, Berlin, Heidelberg, 2004.
URL: https://link.springer.com/chapter/10.1007/978-3-540-28645-5_29
"""


from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector
import numpy as np


class DDM(SuperDetector):

    DETECTOR_NAME = TornadoDic.DDM
    """ Drift Detection Method.

    Parameters
    ----------
    min_num_instances: int (default=30)
        The minimum required number of analyzed samples so change can be 
        detected. This is used to avoid false detections during the early 
        moments of the detector, when the weight of one sample is important.

    warning_level: float (default=2.0)
        Warning Level

    out_control_level: float (default=3.0)
        Out-control Level

    Notes
    -----
    DDM (Drift Detection Method) [1]_ is a concept change detection method
    based on the PAC learning model premise, that the learner's error rate
    will decrease as the number of analysed samples increase, as long as the
    data distribution is stationary.

    If the algorithm detects an increase in the error rate, that surpasses
    a calculated threshold, either change is detected or the algorithm will
    warn the user that change may occur in the near future, which is called
    the warning zone.

    The detection threshold is calculated in function of two statistics,
    obtained when `(pi + si)` is minimum:

    * :math:`p_{min}`: The minimum recorded error rate.
    * `s_{min}`: The minimum recorded standard deviation.

    At instant :math:`i`, the detection algorithm uses:

    * :math:`p_i`: The error rate at instant i.
    * :math:`s_i`: The standard deviation at instant i.

    The conditions for entering the warning zone and detecting change are
    as follows:

    * if :math:`p_i + s_i \geq p_{min} + 2 * s_{min}` -> Warning zone
    * if :math:`p_i + s_i \geq p_{min} + 3 * s_{min}` -> Change detected

    References
    ----------
    .. [1] João Gama, Pedro Medas, Gladys Castillo, Pedro Pereira Rodrigues: Learning
       with Drift Detection. SBIA 2004: 286-295
    """

    def __init__(self, min_num_instances=30, warning_level=2.0, out_control_level=3.0):
        super().__init__()
        self.sample_count = None
        self.miss_prob = None
        self.miss_std = None
        self.miss_prob_sd_min = None
        self.miss_prob_min = None
        self.miss_sd_min = None
        self.in_concept_change = None
        self.in_warning_zone = None
        self.estimation = None
        self.delay = None
        self.min_instances = min_num_instances
        self.warning_level = warning_level
        self.out_control_level = out_control_level
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.sample_count = 1
        self.miss_prob = 1.0
        self.miss_std = 0.0
        self.miss_prob_sd_min = float("inf")
        self.miss_prob_min = float("inf")
        self.miss_sd_min = float("inf")

    def run(self, prediction):
        """ Add a new element to the statistics

        Parameters
        ----------
        prediction: int (either 0 or 1)
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).

        Notes
        -----
        After calling this method, to verify if change was detected or if
        the learner is in the warning zone, one should call the super method
        detected_change, which returns True if concept drift was detected and
        False otherwise.

        """
        if self.in_concept_change:
            self.reset()

        self.miss_prob = self.miss_prob + (prediction - self.miss_prob) / float(self.sample_count)
        self.miss_std = np.sqrt(self.miss_prob * (1 - self.miss_prob) / float(self.sample_count))
        self.sample_count += 1

        self.estimation = self.miss_prob
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0
        if self.sample_count < self.min_instances:
            return False, False

        if self.miss_prob + self.miss_std <= self.miss_prob_sd_min:
            self.miss_prob_min = self.miss_prob
            self.miss_sd_min = self.miss_std
            self.miss_prob_sd_min = self.miss_prob + self.miss_std

        if self.miss_prob + self.miss_std > self.miss_prob_min + self.out_control_level * self.miss_sd_min:
            self.in_concept_change = True

        elif self.miss_prob + self.miss_std > self.miss_prob_min + self.warning_level * self.miss_sd_min:
            self.in_warning_zone = True

        else:
            self.in_warning_zone = False

        return self.in_warning_zone, self.in_concept_change

    def get_settings(self):
        return [str(self.MINIMUM_NUM_INSTANCES),
                "$n_{min}$:" + str(self.MINIMUM_NUM_INSTANCES)]
