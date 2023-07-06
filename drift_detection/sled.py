# Author: Shuxiang Zhang
from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector
import numpy as np


def normalize(item1):
    """
    Normalize false positives and detection delay.
    :param item1: A list of raw values.
    :return: Normalized values.
    """
    item1 = np.array(item1)
    if len(item1) == 1:
        return np.array([1])
    else:
        item1 = item1 / item1.sum()
        item1 = 1 - item1
        item1 = item1 / item1.sum()
        return item1


class Sled(SuperDetector):
    DETECTOR_NAME = TornadoDic.SLED

    def __init__(self, base_detector=None, valid_period=200, detection_threshold=0.5):
        super().__init__()
        self.base_detector = base_detector
        self.final_weights = None
        self.valid_period = valid_period
        self.detection_threshold = detection_threshold
        self.current_time = -1
        self.pred_result = []

    def reset(self):
        super().reset()
        self.pred_result = []
        for i in self.base_detector:
            i.reset()

    def train(self, smoothing_factor, fusion_parameter, drift_acceptance_interval, true_change,
              detected_drifts):
        """
        Method used to train SLED on a training data stream.
        :param smoothing_factor: Parameter used to balance information between training sessions.
        :param fusion_parameter: Parameter used to balance fp and dt at one training session.
        :param drift_acceptance_interval: The furthest data point from the true drift point that is considered a drift.
        :param true_change: The labelled true drifts used to train SLED.
        :param detected_drifts: The drifts detected by different base detectors.
        :return:
        """
        final_weights = None
        fw = []
        start_point = 0
        for t_c in true_change:
            false_positive = dict()
            delay_time = dict()
            flag = dict()
            for i in range(len(self.base_detector)):
                false_positive[i] = 0  # Initialize a dictionary for keeping false positives for each base detector
                delay_time[i] = 0  # Initialize a dictionary for keeping delay time
                flag[i] = False
            for detector in range(len(self.base_detector)):
                for drift in detected_drifts[detector]:
                    if drift in range(start_point, t_c + drift_acceptance_interval):
                        if drift < t_c:
                            false_positive[detector] += 1
                        else:
                            if flag[detector] is False:
                                delay_time[detector] = drift - t_c
                                flag[detector] = True
            start_point = t_c + drift_acceptance_interval
            normalized_false_positive = []
            normalized_delay_time = []
            sorted_flag = []
            for i in range(len(self.base_detector)):  # Rearrange the elements in the order of detector id
                normalized_false_positive.append(false_positive[i])
                normalized_delay_time.append(delay_time[i])
                sorted_flag.append(flag[i])
            failed_detector = []
            for i in range(len(self.base_detector)):
                if not sorted_flag[i]:
                    failed_detector.append(i)
            if len(failed_detector) != 0:
                # Remove values for failed detectors in both false positives and delay time
                nfp = np.array(normalized_false_positive.copy())
                ndt = np.array((normalized_delay_time.copy()))
                diff = set(list(range(len(nfp)))).difference(set(failed_detector))
                diff = np.array(list(diff))
                if len(diff) != 0:
                    nfp = nfp[diff]
                    ndt = ndt[diff]
                    if all(x == 0 for x in nfp):
                        nfp = [1 / len(nfp)] * len(nfp)
                    else:
                        nfp = list(normalize(nfp))
                    if all(x == 0 for x in ndt):
                        ndt = [1 / len(ndt)] * len(ndt)
                    else:
                        ndt = list(normalize(ndt))
                    for i in range(len(self.base_detector)):
                        if i in failed_detector:
                            normalized_false_positive[i] = 0
                            normalized_delay_time[i] = 0
                        else:
                            normalized_false_positive[i] = nfp.pop(0)
                            normalized_delay_time[i] = ndt.pop(0)
                    weight_1 = np.array(normalized_false_positive)
                    weight_2 = np.array(normalized_delay_time)
                else:
                    weight_1 = np.array([1 / len(nfp)] * len(nfp))
                    weight_2 = np.array([1 / len(nfp)] * len(nfp))
            else:
                if all(x == 0 for x in normalized_false_positive):
                    weight_1 = np.array([1 / len(normalized_false_positive)]) * len(normalized_false_positive)
                else:
                    weight_1 = normalize(normalized_false_positive)
                if all(x == 0 for x in normalized_delay_time):
                    weight_2 = np.array([1 / len(normalized_delay_time)]) * len(normalized_delay_time)
                else:
                    weight_2 = normalize(normalized_delay_time)

            current_weight = fusion_parameter * weight_1 + (1 - fusion_parameter) * weight_2
            current_weight = current_weight / current_weight.sum()
            # print(f'Current weights: {current_weight}')
            if final_weights is None and not (self.base_detector is None):
                final_weights = list(current_weight)
            else:
                final_weight = smoothing_factor * np.array(final_weights) + (1 - smoothing_factor) * current_weight
                final_weight = final_weight / final_weight.sum()
                final_weights = list(final_weight)
            fw.append(final_weights)
        self.final_weights = fw[-1]

    def train_ber(self, smoothing_factor, fusion_parameter, drift_acceptance_interval, true_change, concept_length,
                  data_stream):
        """
        Method used to train SLED on a training data stream.
        :param smoothing_factor: Parameter used to balance information between training sessions.
        :param fusion_parameter: Parameter used to balance fp and dt at one training session.
        :param drift_acceptance_interval: The furthest data point from the true drift point that is considered a drift.
        :param true_change: The labelled true drifts used to train SLED.
        :param concept_length: The length of a concept.
        :param data_stream: The training data stream.
        :return:
        """
        final_weights = None
        fw = []
        start_point = 0
        for t_c in true_change:
            for det in self.base_detector:
                det.reset()
            false_positive = dict()
            delay_time = dict()
            flag = dict()
            for i in range(len(self.base_detector)):
                false_positive[i] = 0  # Initialize a dictionary for keeping false positives for each base detector
                delay_time[i] = 0  # Initialize a dictionary for keeping delay time
                flag[i] = False
            for detector in range(len(self.base_detector)):
                for i in range(start_point, t_c + drift_acceptance_interval):
                    drift_status = self.base_detector[detector].run(data_stream[i])[1]
                    # print(f'change was detected by {detector} at index {i}')
                    if drift_status:
                        if i < t_c:
                            false_positive[detector] += 1
                        else:
                            if flag[detector] is False:
                                delay_time[detector] = i - t_c
                                flag[detector] = True
            start_point = t_c + concept_length
            normalized_false_positive = []
            normalized_delay_time = []
            sorted_flag = []
            for i in range(len(self.base_detector)):  # Rearrange the elements in the order of detector id
                normalized_false_positive.append(false_positive[i])
                normalized_delay_time.append(delay_time[i])
                sorted_flag.append(flag[i])
            failed_detector = []
            for i in range(len(self.base_detector)):
                if not sorted_flag[i]:
                    failed_detector.append(i)
            if len(failed_detector) != 0:
                # Remove values for failed detectors in both false positives and delay time
                nfp = np.array(normalized_false_positive.copy())
                ndt = np.array((normalized_delay_time.copy()))
                diff = set(list(range(len(nfp)))).difference(set(failed_detector))
                diff = np.array(list(diff))
                if len(diff) != 0:
                    nfp = nfp[diff]
                    ndt = ndt[diff]
                    if all(x == 0 for x in nfp):
                        nfp = [1 / len(nfp)] * len(nfp)
                    else:
                        nfp = list(normalize(nfp))
                    if all(x == 0 for x in ndt):
                        ndt = [1 / len(ndt)] * len(ndt)
                    else:
                        ndt = list(normalize(ndt))
                    for i in range(len(self.base_detector)):
                        if i in failed_detector:
                            normalized_false_positive[i] = 0
                            normalized_delay_time[i] = 0
                        else:
                            normalized_false_positive[i] = nfp.pop(0)
                            normalized_delay_time[i] = ndt.pop(0)
                    weight_1 = np.array(normalized_false_positive)
                    weight_2 = np.array(normalized_delay_time)
                else:
                    weight_1 = np.array([1 / len(nfp)] * len(nfp))
                    weight_2 = np.array([1 / len(nfp)] * len(nfp))
            else:
                if all(x == 0 for x in normalized_false_positive):
                    weight_1 = np.array([1 / len(normalized_false_positive)]) * len(normalized_false_positive)
                else:
                    weight_1 = normalize(normalized_false_positive)
                if all(x == 0 for x in normalized_delay_time):
                    weight_2 = np.array([1 / len(normalized_delay_time)]) * len(normalized_delay_time)
                else:
                    weight_2 = normalize(normalized_delay_time)

            current_weight = fusion_parameter * weight_1 + (1 - fusion_parameter) * weight_2
            current_weight = current_weight / current_weight.sum()
            # print(f'Current weights: {current_weight}')
            if final_weights is None and not (self.base_detector is None):
                final_weights = list(current_weight)
            else:
                final_weight = smoothing_factor * np.array(final_weights) + (1 - smoothing_factor) * current_weight
                final_weight = final_weight / final_weight.sum()
                final_weights = list(final_weight)
            fw.append(final_weights)
        self.final_weights = fw[-1]

    def run(self, input_value):
        warning_status = False
        drift_status = False
        self.current_time += 1
        for i in range(len(self.base_detector)):
            if self.base_detector[i].run(input_value)[1]:
                self.base_detector[i].reset()
                self.pred_result.append((i, self.current_time))
        new = []
        for a, b in self.pred_result:
            if self.current_time-b >= self.valid_period:
                new.append((a, b))
        new_result = [e for e in self.pred_result if e not in new]
        self.pred_result = new_result
        alarm = 0
        if len(self.pred_result) == 0 or len(self.pred_result) == 1:
            pass
        else:
            detector_id = set()
            for x, y in self.pred_result:
                detector_id.add(x)
            for i in detector_id:
                alarm = alarm + self.final_weights[i]
            print(alarm)
            if alarm >= self.detection_threshold:
                drift_status = True
                return warning_status, drift_status
        return warning_status, drift_status

    def get_settings(self):
        return [str(len(self.base_detector)), "$fp:$"+str(self.current_time)]


