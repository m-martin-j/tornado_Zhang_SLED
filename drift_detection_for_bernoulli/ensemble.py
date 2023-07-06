# Author: Shuxiang Zhang
import numpy as np
from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector
from filters.attribute_handlers import *
from streams.readers.arff_reader import *
import copy
class Ensemble(SuperDetector):
    DETECTOR_NAME = TornadoDic.ENSEMBLE
    UPDATE_INTERVAL = 2000

    def __init__(self, base_detector = None, learning_rate = 0.5, fusion_parameter = 0.8):
        super().__init__()
        self.base_detector = base_detector
        self.learning_rate = learning_rate
        self.fusion_parameter = fusion_parameter
        self.final_weights = None
        self.current_time = 0
        self.pred_result = []
        self.flag = 'a'

    def reset(self):
        super().reset()
        for i in self.base_detector:
            i.reset()

    def training(self, true_drift_points, stream, pretrain_stream, learner):
        false_positive = dict()
        delay_time = dict()
        check_point = dict()
        result_set = []
        for i in range(len(self.base_detector)):
            instance_counter = 0
            learner.reset()
            # Read the stored data dream and use the first 200 instances to train the classifier
            X = []
            Y = []
            for j in range(len(pretrain_stream)):
                X.append(pretrain_stream[j][:2])
                y = pretrain_stream[j][-1]
                Y.append(y)
            X = np.array(X)
            Y = np.array(Y)
            learner.partial_fit(X, Y, classes=['p', 'n'])
            # Run the trained classifier on the remaining 100000 instances and make predictions
            for l in range(len(stream)):
                instance_counter += 1
                x = stream[l][:2]
                x = np.array(x)
                x = x.reshape(1, -1)
                y = stream[l][-1]
                y = np.array(y)
                y = y.reshape(1, -1)
                predicted_y = learner.predict(x)
                prediction_status = True if predicted_y == y else False
                warning_status, drift_status = self.base_detector[i].detect(prediction_status)
                if drift_status:
                    print(f'Change was detected at by {self.base_detector[i].DETECTOR_NAME} at index {instance_counter}')
                    result_set.append((i, instance_counter))
                    learner.reset()
                    self.base_detector[i].reset()
                y = y[0]
                learner.partial_fit(x, y, classes=['p', 'n'])

        index = 0
        for t in true_drift_points:
            for i in range(len(self.base_detector)):
                false_positive[i] = 0  # Initialize a dictionary for keeping false positives for each base detector
                delay_time[i] = 0  # Initialize a dictionary for keeping delay time
                check_point[i] = False
            for r in result_set:
                if r[1] >= index and r[1] < t:
                    false_positive[r[0]] += 1
                if r[1] >= t and r[1] <= t + Ensemble.UPDATE_INTERVAL:
                    if not check_point[r[0]]:
                        delay_time[r[0]] = r[1] - t
                        check_point[r[0]] = True
                    else:
                        false_positive[r[0]] += 1
            normalized_false_positive = []
            normalized_delay_time = []
            sorted_check_point = []
            for i in range(len(self.base_detector)): # Rearrange the elements in the order of detector id
                normalized_false_positive.append(false_positive[i])
                normalized_delay_time.append(delay_time[i])
                sorted_check_point.append(check_point[i])
            failed_detector = []
            for i in range(len(self.base_detector)):
                if not sorted_check_point[i]:
                    failed_detector.append(i)
            if len(failed_detector) != 0:
                # Remove values for failed detectors in both false positives and delay time
                nfp = np.array(normalized_false_positive.copy())
                ndt = np.array((normalized_delay_time.copy()))
                diff = set(list(range(len(nfp)))).difference(set(failed_detector))
                diff = np.array(list(diff))
                nfp = nfp[diff]
                ndt = ndt[diff]
                if all(x == 0 for x in nfp):
                    nfp = [1/len(nfp)]*len(nfp)
                else:
                    nfp = list(self.normalize(nfp))
                if all(x == 0 for x in ndt):
                    ndt = [1/len(ndt)]*len(ndt)
                else:
                    ndt = list(self.normalize(ndt))
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
                if all(x == 0 for x in normalized_false_positive):
                    weight_1 = np.array([1/len(normalized_false_positive)])*len(normalized_false_positive)
                else:
                    weight_1 = self.normalize(normalized_false_positive)
                if all(x == 0 for x in normalized_delay_time):
                    weight_2 =  np.array([1/len(normalized_delay_time)])*len(normalized_delay_time)
                else:
                    weight_2 = self.normalize(normalized_delay_time)

            current_weight = self.fusion_parameter * weight_1 + (1 - self.fusion_parameter) * weight_2
            current_weight = current_weight/current_weight.sum()

            if self.final_weights is None and not (self.base_detector is None):
                self.final_weights = list(current_weight)
            else:
                final_weight = self.learning_rate * np.array(self.final_weights) + (1 - self.learning_rate) * current_weight
                final_weight = final_weight/final_weight.sum()
                self.final_weights = list(final_weight)

            # if self.old_final_weights is None and not (self.base_detector is None):
            #     self.old_final_weights = list(current_weight)
            # else:
            #     final_weight = self.learning_rate * np.array(self.old_final_weights) + (1 - self.learning_rate) * current_weight
            #     final_weight = final_weight/final_weight.sum()
            #     self.old_final_weights = list(final_weight)
            #
            # magnitude_of_average = self.magnitude(1/len(self.old_final_weights))
            # for w in range(len(self.old_final_weights)):
            #     if self.old_final_weights[w] == 0:
            #         removed_detectors.append(w)
            #     elif self.magnitude(self.old_final_weights[w]) > magnitude_of_average:
            #         removed_detectors.append(w)
            # if len(removed_detectors) != 0:
            #     old = np.array(self.old_final_weights.copy())
            #     diff = set(list(range(len(old)))).difference(set(removed_detectors))
            #     for i in removed_detectors:
            #         old[i] = 0
            #     diff = np.array(list(diff))
            #     remaining_detector = old[diff]
            #     re_weight = self.normalize(remaining_detector)
            #     old = list(old)
            #     diff = list(diff)
            #     re_weight = list(re_weight)
            #     for i in range(len(diff)):
            #         old[diff[i]] = re_weight[i]
            #     self.final_weights = old
            # else:
            #     self.final_weights = self.old_final_weights
            index = index + t
            # Reset all base detectors
            for i in range(len(self.base_detector)):
                self.base_detector[i].reset()

    @staticmethod
    def normalize(item):
        item = np.array(item)
        if len(item) == 1:
            return np.array([1])
        else:
            item = item / item.sum()
            item = 1 - item
            item = item / item.sum()
            return item

    def run(self, input_value):
        warning_status = False
        drift_status = False
        self.current_time += 1
        for i in range(len(self.base_detector)):
            if self.base_detector[i].run(input_value)[1]:
                print(f'Change was detected by {self.base_detector[i].DETECTOR_NAME} at index {self.current_time}')
                self.pred_result.append((i, self.current_time))
                self.base_detector[i].reset()
        alarm = 0
        if len(self.pred_result) == 0 or len(self.pred_result) == 1:
            pass
        #elif self.pred_result[-1][1] - self.pred_result[0][1] >= self.window_size:
        else:
            detector_id = set()
            for x, y in self.pred_result:
                detector_id.add(x)
            for i in detector_id:
                alarm = alarm + self.final_weights[i]
            if alarm >= 0.3:
                drift_status = True
                self.pred_result = []
                return warning_status,drift_status
            # remove the oldest element
            # new = []
            # first = self.pred_result[0][1]
            # for i,j in self.pred_result:
            #     if j != first:
            #         new.append((i,j))
            # self.pred_result = new
        return warning_status,drift_status

    def __len__(self):
        return len(self.flag)

    def get_settings(self):
        return [str(self.learning_rate) + '.' + "$fp:$" + str(self.fusion_parameter)]


