root_name = 'tnseq_adaptation'
import os, sys  ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)
from methods.config import *
import numpy as np
import pickle

with open(f'{data_dir}/bac_lineage_positions.pkl', 'rb') as f:
    bac_lineage_positions = pickle.load(f)

with open(f'{data_dir}/bac_row_ids.pkl', 'rb') as f:
    bac_row_ids = pickle.load(f)

with open(f'{data_dir}/bac_read_arrays.pkl', 'rb') as f:
    bac_read_arrays = pickle.load(f)

with open(f'{data_dir}/bac_input_arrays.pkl', 'rb') as f:
    bac_input_arrays = pickle.load(f)

for bac in BACTERIA:
    row_id_tuples = sorted([row, id] for id, row in bac_row_ids[bac].items())
    with open(f'{data_dir}/{bac}_read_arrays.txt', 'w') as f:
        header = 'mouse\tday\t' + '\t'.join([str(e) for e in range(bac_read_arrays[bac].shape[1])]) + '\n'
        f.write(header)
        positions = bac_lineage_positions[bac]
        positions_line = 'lineage\tpositions\t' + '\t'.join([str(e) for e in bac_lineage_positions[bac]]) + '\n'
        f.write(positions_line)

        for row, id in row_id_tuples:
            mouse, day = id[0], id[1]
            read_vec = bac_read_arrays[bac][row]
            row_items = [str(mouse), str(day)] + [str(e) for e in read_vec]
            line = '\t'.join(row_items) + '\n'
            f.write(line)

        for row, read_vec in enumerate(bac_input_arrays[bac]):
            mouse, rep_num = 'input', row+1
            row_items = [str(mouse), str(rep_num)] + [str(e) for e in read_vec]
            line = '\t'.join(row_items) + '\n'
            f.write(line)

