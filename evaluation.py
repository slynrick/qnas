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
        gpus = tf.config.list_logical_devices('GPU')
        if gpus:
            selected_gpu = 0
            individual_per_gpu = []
            for idx in range(len(variables)):
                self.logger.info(f"Going to start fitness of individual {idx} on gpu {selected_gpu} ({gpus[selected_gpu].name})")
                individual_per_gpu.append((idx, selected_gpu, decoded_nets[idx], decoded_params[idx], variables[idx]))
                selected_gpu += 1
                if selected_gpu >= len(gpus):
                    selected_gpu = selected_gpu = selected_gpu % len(gpus)
            
            
            processes = []
            for idx, gpu in enumerate(gpus):
                individuals_selected_gpu = list(filter(lambda x: x[1]==idx, individual_per_gpu))
                print(individuals_selected_gpu)
                process = Process(target=self.run_individuals, args=(generation,
                                                    self.data_info,
                                                    self.train_params,
                                                    self.fn_dict,
                                                    gpu.name, 
                                                    individuals_selected_gpu))
                process.start()
                processes.append(process)

            for p in processes:
                p.join()
                    
        for idx, val in enumerate(variables):
            evaluations[idx] = val.value
        

        return evaluations



    def run_individuals(self, generation, data_info, train_params, fn_dict, selected_gpu, individuals_selected_gpu):
        for individual, selected_gpu_id, decoded_net, decoded_params, return_val in individuals_selected_gpu:
            print(f"starting individual {individual}")
            train.fitness_calculation(f"{generation}_{selected_gpu_id}_{individual}",
                                        data_info,
                                        {**train_params,**decoded_params},
                                        fn_dict,
                                        decoded_net, 
                                        selected_gpu, 
                                        return_val)
            print(f"finishing individual {individual} - {return_val.value}")
            self.logger.info(f"Clculated fitness of individual {individual} on gpu {selected_gpu} with {return_val.value}")

