"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import copy
import random

import numpy
from pympler import asizeof

from archiver.archiver import Archiver
from evaluators.classifier_evaluator import PredictionEvaluator
from evaluators.detector_evaluator import DriftDetectionEvaluator
from plotter.performance_plotter import *
from filters.attribute_handlers import *
from streams.readers.arff_reader import *


class PrequentialDriftEvaluator_real_world:
    """This class lets one run a classifier with a drift detector against a data stream,
    and evaluate it prequentially over time. Also, one is able to measure the detection
    false positive as well as false negative rates."""

    def __init__(self, learner, drift_detector, attributes, attributes_scheme, project, memory_check_step=-1):
        self.drift_order = 0
        self.learner = learner
        self.drift_detector = drift_detector
        self.instance_counter = 0
        self.__num_rubbish = 0
        self.__learner_error_rate_array = []
        self.__learner_memory_usage = []
        self.__learner_runtime = []
        self.located_drift_points = []
        self.__drift_points_boolean = []
        self.__drift_detection_memory_usage = []
        self.__drift_detection_runtime = []
        self.__attributes = attributes
        self.__numeric_attribute_scheme = attributes_scheme['numeric']
        self.__nominal_attribute_scheme = attributes_scheme['nominal']

        self.__project_path = project.get_path()
        self.__project_name = project.get_name()

        self.__memory_check_step = memory_check_step

    def run_1(self, stream, located_drifts, drift_location, random_seed=1):
        random.seed(random_seed)
        num_drifts = 0
        for record in stream:
            self.instance_counter += 1
            percentage = (self.instance_counter / len(stream)) * 100
            print("%0.2f" % percentage + "% of instances are prequentially processed!", end="\r")

            if record.__contains__("?"):
                self.__num_rubbish += 1
                continue

            # ---------------------
            #  Data Transformation
            # ---------------------
            r = copy.copy(record)
            for k in range(0, len(r) - 1):
                if self.learner.LEARNER_CATEGORY == TornadoDic.NOM_CLASSIFIER and self.__attributes[k].TYPE == TornadoDic.NUMERIC_ATTRIBUTE:
                    r[k] = Discretizer.find_bin(r[k], self.__nominal_attribute_scheme[k])
                elif self.learner.LEARNER_CATEGORY == TornadoDic.NUM_CLASSIFIER and self.__attributes[k].TYPE == TornadoDic.NOMINAL_ATTRIBUTE:
                    r[k] = NominalToNumericTransformer.map_attribute_value(r[k], self.__numeric_attribute_scheme[k])
            # NORMALIZING NUMERIC DATA
            if self.learner.LEARNER_CATEGORY == TornadoDic.NUM_CLASSIFIER:
                r[0:len(r) - 1] = Normalizer.normalize(r[0:len(r) - 1], self.__numeric_attribute_scheme)

            # ----------------------
            #  Prequential Learning
            # ----------------------
            if self.learner.is_ready():

                real_class = r[len(r) - 1]
                predicted_class = self.learner.do_testing(r)

                prediction_status = True
                if real_class != predicted_class:
                    prediction_status = False

                # -----------------------
                #  Drift Detected?
                # -----------------------
                if self.instance_counter in located_drifts:
                    num_drifts += 1
                    self.__drift_points_boolean.append(1)
                    self.located_drift_points.append(self.instance_counter)

                    learner_error_rate = PredictionEvaluator.calculate(TornadoDic.ERROR_RATE,
                                                                       self.learner.get_global_confusion_matrix())
                    self.__learner_error_rate_array.append(round(learner_error_rate, 4))
                    self.__learner_memory_usage.append(asizeof.asizeof(self.learner, limit=20))
                    self.__learner_runtime.append(self.learner.get_running_time())

                    self.__drift_detection_memory_usage.append(asizeof.asizeof(self.drift_detector, limit=20))
                    self.__drift_detection_runtime.append(self.drift_detector.RUNTIME)

                    if num_drifts < drift_location:
                        self.learner.reset()
                    elif num_drifts == drift_location:
                        pass
                    else:
                        break

                    continue

                if self.learner.LEARNER_TYPE == TornadoDic.TRAINABLE:
                    self.learner.do_training(r)
                else:
                    self.learner.do_loading(r)
            else:
                if self.learner.LEARNER_TYPE == TornadoDic.TRAINABLE:
                    self.learner.do_training(r)
                else:
                    self.learner.do_loading(r)

                self.learner.set_ready()
                self.learner.update_confusion_matrix(r[len(r) - 1], r[len(r) - 1])

            # learner_error_rate = PredictionEvaluator.calculate(TornadoDic.ERROR_RATE,
            #                                                    self.learner.get_confusion_matrix())
            # learner_error_rate = round(learner_error_rate, 4)
            # self.__learner_error_rate_array.append(learner_error_rate)

            if self.__memory_check_step != -1:
                if self.instance_counter % self.__memory_check_step == 0:
                    self.__drift_detection_memory_usage.append(asizeof.asizeof(self.drift_detector, limit=20))

            self.__drift_points_boolean.append(0)

        # lrn_error_rate = PredictionEvaluator.calculate_error_rate(self.learner.get_global_confusion_matrix())

        return self.located_drift_points,self.__learner_error_rate_array

    def run_2(self, stream, random_seed=1):

        random.seed(random_seed)

        for record in stream:

            self.instance_counter += 1

            if record.__contains__("?"):
                self.__num_rubbish += 1
                continue

            # ---------------------
            #  Data Transformation
            # ---------------------
            r = copy.copy(record)
            for k in range(0, len(r) - 1):
                if self.learner.LEARNER_CATEGORY == TornadoDic.NOM_CLASSIFIER and self.__attributes[k].TYPE == TornadoDic.NUMERIC_ATTRIBUTE:
                    r[k] = Discretizer.find_bin(r[k], self.__nominal_attribute_scheme[k])
                elif self.learner.LEARNER_CATEGORY == TornadoDic.NUM_CLASSIFIER and self.__attributes[k].TYPE == TornadoDic.NOMINAL_ATTRIBUTE:
                    r[k] = NominalToNumericTransformer.map_attribute_value(r[k], self.__numeric_attribute_scheme[k])
            # NORMALIZING NUMERIC DATA
            if self.learner.LEARNER_CATEGORY == TornadoDic.NUM_CLASSIFIER:
                r[0:len(r) - 1] = Normalizer.normalize(r[0:len(r) - 1], self.__numeric_attribute_scheme)

            # ----------------------
            #  Prequential Learning
            # ----------------------
            if self.learner.is_ready():

                real_class = r[len(r) - 1]
                predicted_class = self.learner.do_testing(r)

                prediction_status = True
                if real_class != predicted_class:
                    prediction_status = False

                # -----------------------
                #  Drift Detected?
                # -----------------------
                warning_status, drift_status = self.drift_detector.detect(prediction_status)
                if drift_status:
                    self.__drift_points_boolean.append(1)
                    self.located_drift_points.append(self.instance_counter)

                    learner_error_rate = PredictionEvaluator.calculate(TornadoDic.ERROR_RATE,
                                                                       self.learner.get_global_confusion_matrix())
                    self.__learner_error_rate_array.append(round(learner_error_rate, 4))
                    self.__learner_memory_usage.append(asizeof.asizeof(self.learner, limit=20))
                    self.__learner_runtime.append(self.learner.get_running_time())

                    self.__drift_detection_memory_usage.append(asizeof.asizeof(self.drift_detector, limit=20))
                    self.__drift_detection_runtime.append(self.drift_detector.RUNTIME)
                    self.learner.reset()
                    self.drift_detector.reset()

                    continue

                if self.learner.LEARNER_TYPE == TornadoDic.TRAINABLE:
                    self.learner.do_training(r)
                else:
                    self.learner.do_loading(r)
            else:
                if self.learner.LEARNER_TYPE == TornadoDic.TRAINABLE:
                    self.learner.do_training(r)
                else:
                    self.learner.do_loading(r)

                self.learner.set_ready()
                self.learner.update_confusion_matrix(r[len(r) - 1], r[len(r) - 1])

            # learner_error_rate = PredictionEvaluator.calculate(TornadoDic.ERROR_RATE,
            #                                                    self.learner.get_confusion_matrix())
            # learner_error_rate = round(learner_error_rate, 4)
            # self.__learner_error_rate_array.append(learner_error_rate)

            if self.__memory_check_step != -1:
                if self.instance_counter % self.__memory_check_step == 0:
                    self.__drift_detection_memory_usage.append(asizeof.asizeof(self.drift_detector, limit=20))

            self.__drift_points_boolean.append(0)

        # lrn_error_rate = PredictionEvaluator.calculate_error_rate(self.learner.get_global_confusion_matrix())

        return self.located_drift_points,self.__learner_error_rate_array