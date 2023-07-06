import numpy as np
from streams.generators.tools.transition_functions import Transition


class Hyperplane:
    """ HyperplaneGenerator
    Generates a problem of prediction class of a rotation hyperplane. It was
    used as testbed for CVFDT and VFDT in [1]_.

    A hyperplane in d-dimensional space is the set of points x that satisfy
    :math:`\sum^{d}_{i=1} w_i x_i = w_0 = \sum^{d}_{i=1} w_i` where
    :math:`x_i`, is the ith coordinate of x. Examples for which
    :math:`\sum^{d}_{i=1} w_i x_i > w_0` are labeled positive, and examples
    for which :math:`\sum^{d}_{i=1} w_i x_i \leq w_0` are labeled negative.

    Hyperplanes are useful for simulation time-changing concepts, because we
    can change the orientation and position of the hyperplane ina  smooth
    manner by changing the relative size of the weights. We introduce change
    to this dataset adding drift to each weight feature :math:`w_i = w_i + d \sigma`,
    where :math:`\sigma` is the probability that the direction of change is
    reversed and d is the change applied to every example.

    Parameters
    ----------
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_features: int (Default 10)
        The number of attributes to generate.
        Higher than 2.

    n_drift_features: int (Default: 2)
        The number of attributes with drift.
        Higher than 2.

    mag_change: float (Default: 0.0)
        Magnitude of the change for every example.
        From 0.0 to 1.0.

    noise_percentage: float (Default: 0.05)
        Percentage of noise to add to the data.
        From 0.0 to 1.0.

    sigma_percentage: int (Default 0.1)
        Percentage of probability that the direction of change is reversed.
        From 0.0 to 1.0.

    References
    ----------
    .. [1] G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
       In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.

    """

    def __init__(self, concept_length,random_seed = 1, n_features=10, transition_length=500,n_drift_features=2, mag_change=None,
                 noise_percentage=0.1, sigma_percentage=0.1):
        super().__init__()

        self.n_num_features = n_features
        self.__CONCEPT_LENGTH = concept_length
        self.__INSTANCES_NUM = concept_length*len(mag_change)
        self.__NUM_DRIFTS = len(mag_change)-1
        self.n_features = self.n_num_features
        self.__RECORDS = []
        self.__W = transition_length
        self.n_classes = 2
        self.n_drift_features = n_drift_features
        self.mag_change = mag_change
        self.sigma_percentage = sigma_percentage
        self.noise_percentage = noise_percentage
        self.n_targets = 1
        self.__RANDOM_SEED = random_seed
        self._next_class_should_be_zero = False
        self._weights = np.zeros(self.n_features)
        self._sigma = np.zeros(self.n_features)
        self.name = "Hyperplane"

        self.__configure()
        self.initialize_record()
        print("You are going to generate a " + self.name + " data stream containing " +
              str(self.__INSTANCES_NUM) + " instances, and " + str(self.__NUM_DRIFTS) + " concept drifts; " + "\n\r" +
              "where they appear at every " + str(self.__CONCEPT_LENGTH) + " instances.")

    def __configure(self):
        self.target_names = ["target_0"]
        self.feature_names = ["att_num_" + str(i) for i in range(self.n_features)]
        self.target_values = [i for i in range(self.n_classes)]

    @property
    def n_drift_features(self):
        """ Retrieve the number of drift features.

        Returns
        -------
        int
            The total number of drift features.

        """
        return self._n_drift_features

    @n_drift_features.setter
    def n_drift_features(self, n_drift_features):
        """ Set the number of drift features

        """
        self._n_drift_features = n_drift_features

    @property
    def noise_percentage(self):
        """ Retrieve the value of the value of Noise percentage

        Returns
        -------
        float
            percentage of the noise
        """
        return self._noise_percentage

    @noise_percentage.setter
    def noise_percentage(self, noise_percentage):
        """ Set the value of the value of noise percentage.

        Parameters
        ----------
        noise_percentage: float (0.0..1.0)

        """
        if (0.0 <= noise_percentage) and (noise_percentage <= 1.0):
            self._noise_percentage = noise_percentage
        else:
            raise ValueError("noise percentage should be in [0.0..1.0], {} was passed".format(noise_percentage))

    @property
    def sigma_percentage(self):
        """ Retrieve the value of the value of sigma percentage

        Returns
        -------
        float
            percentage of the sigma
        """
        return self._sigma_percentage

    @sigma_percentage.setter
    def sigma_percentage(self, sigma_percentage):
        """ Set the value of the value of noise percentage.

        Parameters
        ----------
        sigma_percentage: float (0.0..1.0)

        """
        if (0.0 <= sigma_percentage) and (sigma_percentage <= 1.0):
            self._sigma_percentage = sigma_percentage
        else:
            raise ValueError("sigma percentage should be in [0.0..1.0], {} was passed".format(sigma_percentage))

    def generate(self, output_path="HYPERPLANE"):

        np.random.seed(self.__RANDOM_SEED)

        # [1] CREATING RECORDS
        for i in range(0, self.__INSTANCES_NUM):
            concept_sec = int(i / self.__CONCEPT_LENGTH)
            record = self.create_record(self.mag_change[concept_sec],batch_size=1)
            self.__RECORDS.append(list(record))
        if self.__W > 0:
            # [2] TRANSITION
            for i in range(0, self.__NUM_DRIFTS):
                transition = []
                for j in range(0, self.__W):
                    if np.random.random() < Transition.sigmoid(j, self.__W):
                        record = self.create_record(self.mag_change[i + 1],batch_size=1)
                    else:
                        record = self.create_record(self.mag_change[i],batch_size=1)
                    transition.append(list(record))
                starting_index = (i+1) * self.__CONCEPT_LENGTH
                ending_index = starting_index + self.__W
                self.__RECORDS[starting_index: ending_index] = transition
        self.write_to_arff(output_path + ".arff")


    def initialize_record(self):
        self._next_class_should_be_zero = False
        for i in range(self.n_features):
            self._weights[i] = np.random.rand()
            self._sigma[i] = 1 if (i < self.n_drift_features) else 0


    def create_record(self, mag, batch_size=1):
        """
        Should be called before generating the samples.

        """
        data = np.zeros([batch_size, self.n_features + 1])
        sum_a = 0
        sum_weights = 0.0
        for j in range(batch_size):
            for i in range(self.n_features):
                data[j, i] = np.random.rand()
                sum_a += self._weights[i] * data[j, i]
                sum_weights += self._weights[i]

            group = 1 if sum_a >= sum_weights * 0.5 else 0

            if 0.01 + np.random.rand() <= self.noise_percentage:
                group = 1 if (group == 0) else 0

            data[j, -1] = group

        self._generate_drift(mag)

        current_sample_x = data[:, :self.n_features]
        current_sample_y = data[:, self.n_features:].flatten()
        x = list(current_sample_x[0])
        y = 'p' if list(current_sample_y)[0] == 1 else 'n'
        x.append(y)
        result = tuple(x)
        return result

    def _generate_drift(self,mag):
        """
        Generate drift in the stream.

        """
        for i in range(self.n_drift_features):
            self._weights[i] += float(float(self._sigma[i]) * float(mag))
            if (0.01 + np.random.rand()) <= self.sigma_percentage:
                self._sigma[i] *= -1

    def write_to_arff(self, output_path):

        arff_writer = open(output_path, "w")
        arff_writer.write("@relation HYPERPLANE" + "\n")
        arff_writer.write("@attribute a real" + "\n" +
                          "@attribute b real" + "\n" +
                          "@attribute c real" + "\n" +
                          "@attribute d real" + "\n" +
                          "@attribute e real" + "\n" +
                          "@attribute f real" + "\n" +
                          "@attribute g real" + "\n" +
                          "@attribute h real" + "\n" +
                          "@attribute i real" + "\n" +
                          "@attribute j real" + "\n" +
                          "@attribute class {p,n}" + "\n\n")
        arff_writer.write("@data" + "\n")

        for i in range(0, len(self.__RECORDS)):
            arff_writer.write(str("%0.5f" % self.__RECORDS[i][0]) + "," +
                              str("%0.5f" % self.__RECORDS[i][1]) + "," +
                              str("%0.5f" % self.__RECORDS[i][2]) + "," +
                              str("%0.5f" % self.__RECORDS[i][3]) + "," +
                              str("%0.5f" % self.__RECORDS[i][4]) + "," +
                              str("%0.5f" % self.__RECORDS[i][5]) + "," +
                              str("%0.5f" % self.__RECORDS[i][6]) + "," +
                              str("%0.5f" % self.__RECORDS[i][7]) + "," +
                              str("%0.5f" % self.__RECORDS[i][8]) + "," +
                              str("%0.5f" % self.__RECORDS[i][9]) + "," +
                              self.__RECORDS[i][10] + "\n")
        arff_writer.close()
        print("You can find the generated files in " + output_path + "!")

