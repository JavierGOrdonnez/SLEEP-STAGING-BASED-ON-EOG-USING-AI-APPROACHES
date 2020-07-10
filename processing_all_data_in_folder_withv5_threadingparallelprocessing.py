# list all files in my folders and process them

import os
import sys
sys.path.append('..')
from myfunctions.process_database_data_v5 import process_database_data
import threading
import time

# new line

if __name__ == "__main__":
    t_start = time.time()
    num_threads = 4

    hypnogram_path = 'C:/Users/Equipo/shhs/polysomnography/annotations-events-profusion/shhs1'
    PSG_path = 'C:/Users/Equipo/shhs/polysomnography/edfs/shhs1'

    # target_path = 'C:/Users/Equipo/shhs/polysomnography/processed_data/'
    target_path = 'D:/GitHub/TFM/full_processed_data_v5/'

    PSG_list = [(entry.name, entry.path) for entry in os.scandir(PSG_path)]
    hypnogram_list = [(entry.name, entry.path) for entry in os.scandir(hypnogram_path)]
    lenh = len(hypnogram_list)

    print(len(PSG_list))
    print(PSG_list[:3])
    i = 0
    while i < len(PSG_list):  # for each PSG file
        jobs = []
        for t in range(num_threads):
            name = PSG_list[i][0][:12]

            # search for this name in the hypnogram folder
            j = 0
            while j < lenh:
                if hypnogram_list[j][0][:12] == name:
                    # call function to load data, process it and save it in target path
                    thread = threading.Thread(target=process_database_data, args=(PSG_list[i][1],hypnogram_list[j][1],target_path,True,"float32",
                                                                                  False)) # no reprocessing
                    jobs.append(thread)
                    name = PSG_list[i][1][-16:-4]
                    j = lenh  # exit the while(), matching hypnogram already found

                else:
                    j += 1
            i += 1

        for job in jobs:
            job.start()
            print('Job %d started' %i)


        for job in jobs:
            job.join()
            print('Job %d finished' % i)

    # print('Total time for 20 files: %.2f' %(time.time()-t_start))
