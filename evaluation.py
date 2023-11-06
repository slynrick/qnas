""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute population eval using MPI.
"""

from multiprocessing import Process, Value
import time
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from cnn import train
from util import init_log
import random

class EvalPopulation(object):
    def __init__(self, params, data_info, fn_dict, log_level='INFO'):
        """ Initialize EvalPopulation.

        Args:
            params: dictionary with parameters.
            data_info: one of input.*Info objects.
            fn_dict: dict with definitions of the functions (name and parameters);
                format --> {'fn_name': ['FNClass', {'param1': value1, 'param2': value2}]}.
            log_level: (str) one of "INFO", "DEBUG" or "NONE".
        """

        self.train_params = params
        self.data_info = data_info
        self.fn_dict: Dict[Any,Any] = fn_dict
        self.logger = init_log(log_level, name=__name__)

    def __call__(self, decoded_params, decoded_nets, generation):
        """ Train and evaluate *decoded_nets* using the parameters defined in *decoded_params*.

        Args:
            decoded_params: list containing the dict of values of evolved parameters
                (size = num_individuals).
            decoded_nets: list containing the lists of network layers descriptions
                (size = num_individuals).
            generation: (int) generation number.

        Returns:
            numpy array containing evaluations results of each model in *net_lists*.
        """

        pop_size = len(decoded_nets)

        evaluations = np.empty(shape=(pop_size,))

        print(self.fn_dict)

        variables = [Value('f', 0.0) for _ in range(pop_size)]
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            selected_gpu = 0
            processes = []
            for idx in range(len(variables)):
                process = Process(target=train.fitness_calculation, args=(f'{generation}_0',
                                                    self.data_info,
                                                    {**self.train_params,**decoded_params[0]},
                                                    self.fn_dict,
                                                    decoded_nets[0], 
                                                    str(selected_gpu), 
                                                    variables[idx],))
                process.start()
                self.logger.info(f"Starting fitness of individual {idx} on gpu {selected_gpu}")
                processes.append(process)
                selected_gpu += 1
                if selected_gpu >= len(gpus):
                    selected_gpu = selected_gpu = selected_gpu % len(gpus)

            for p in processes:
                p.join()
                    
        for idx, val in enumerate(variables):
            evaluations[idx] = val.value
        

        return evaluations