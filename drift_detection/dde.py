# Author: Shuxiang Zhang
from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class DDE(SuperDetector):
    DETECTOR_NAME = TornadoDic.DDE

    def __init__(self, base_detector=None, sens=1):
        super().__init__()
        self.base_detector = base_detector
        self.count = -1
        self.sens = sens
        self.MaxWait = 100
        self.numWait = dict()
        self.resp = dict()
        for i in range(len(self.base_detector)):
            self.resp[i] = 'stable'
            self.numWait[i] = 0
        if sens == 1:
            self.DETECTOR_NAME = 'DDE_1'
        elif sens == 2:
            self.DETECTOR_NAME = 'DDE_2'
        else:
            self.DETECTOR_NAME = 'DDE_3'

    def reset(self):
        super().reset()
        self.numDrift = 0
        self.numWarning = 0
        for i in range(len(self.base_detector)):
            self.base_detector[i].reset()
            self.resp[i] = 'stable'
            self.numWait[i] = 0
            self.base_detector[i].reset()

    def run(self, input_value):
        self.count += 1
        warning_status = False
        drift_status = False
        numDrift = 0
        numWarning = 0
        for i in range(len(self.base_detector)):
            warning, drift = self.base_detector[i].run(input_value)
            if drift:
                aux = 'drift'
                # print(f'change was detected at index {self.count} by {i}')
                self.base_detector[i].reset()
            elif warning:
                aux = 'warning'
            else:
                aux = 'stable'
            if self.resp[i] != 'drift' or self.numWait[i] > self.MaxWait:
                self.resp[i] = aux
                self.numWait[i] = 0
            if self.resp[i] == 'drift':
                numDrift += 1
                self.numWait[i] += 1
            elif self.resp[i] == 'warning':
                numWarning += 1
        if numDrift >= self.sens:
            drift_status = True
        elif numWarning + numDrift >= self.sens:
            warning_status = True
        return warning_status,drift_status

    def get_settings(self):
        return [str(len(self.base_detector)), "$fp:$"+str(self.count)]


