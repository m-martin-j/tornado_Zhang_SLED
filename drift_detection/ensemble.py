# Author: Shuxiang Zhang
from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class Ensemble(SuperDetector):
    DETECTOR_NAME = TornadoDic.ENSEMBLE

    def __init__(self, base_detector=None):
        super().__init__()
        self.base_detector = base_detector
        self.final_weights = None
        self.current_time = -1
        self.pred_result = []

    def reset(self):
        super().reset()
        self.pred_result = []
        for i in self.base_detector:
            i.reset()

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
            if self.current_time-b >= 200:
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
            if alarm >= 0.5:
                drift_status = True
                return warning_status, drift_status
        return warning_status, drift_status

    def get_settings(self):
        return [str(len(self.base_detector)), "$fp:$"+str(self.current_time)]


