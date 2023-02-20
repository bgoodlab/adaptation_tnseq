root_name = 'tnseq_adaptation'
import os, sys  ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)
from methods.config import *
import numpy as np
import pickle

bac_lineage_positions = {}
bac_read_arrays = {}
bac_input_arrays = {}
bac_row_ids = {bac:{} for bac in BACTERIA}

for bac in BACTERIA:
    with open(f'{data_dir}/{bac}_read_arrays.txt', 'r') as f:
        header = next(f)
        lineage_positions_line_items = next(f).rstrip('\n').split('\t')
        lineage_positions = np.array([int(e) for e in lineage_positions_line_items[2:]])
        bac_lineage_positions[bac] = lineage_positions

        read_array = []
        input_array = []

        for row, line in enumerate(f):
            line_items = line.rstrip('\n').split('\t')
            lineage_reads = [int(e) for e in line_items[2:]]

            if line_items[0] == 'input' and int(line_items[1]) != 0:
                id = ('input', int(line_items[1]))
                input_array.append(lineage_reads)
                continue
            elif line_items[0] in ['input'] + MEDIA: #input or in vitro
                id = (line_items[0], int(line_items[1]))
                read_array.append(lineage_reads)
            else:
                id = (int(line_items[0]), int(line_items[1]))
                read_array.append(lineage_reads)
            bac_row_ids[bac][id] = row
        bac_read_arrays[bac] = np.array(read_array)
        bac_input_arrays[bac] = np.array(input_array)

with open(f'{data_dir}/bac_lineage_positions.pkl', 'wb') as f:
    pickle.dump(bac_lineage_positions, f)

with open(f'{data_dir}/bac_row_ids.pkl', 'wb') as f:
    pickle.dump(bac_row_ids, f)

with open(f'{data_dir}/bac_read_arrays.pkl', 'wb') as f:
    pickle.dump(bac_read_arrays, f)

with open(f'{data_dir}/bac_input_arrays.pkl', 'wb') as f:
    pickle.dump(bac_input_arrays, f)
