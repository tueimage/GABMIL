
import argparse
from calflops import calculate_flops # type: ignore
from utils.utils import *
from models.model_gabmil import GABMIL
from models.model_transmil import TransMIL


# Generic settings
parser = argparse.ArgumentParser(description='Configurations for WSI Evaluation')
parser.add_argument('--use_local', action='store_true', default=False, help='no global information')
parser.add_argument('--use_grid', action='store_true', default=False, help='enable grid information')
parser.add_argument('--use_block', action='store_true', default=False, help='enable block information')
parser.add_argument('--win_size_b', type=int, default=1, help='block window size')
parser.add_argument('--win_size_g', type=int, default=1, help='grid window size')
parser.add_argument('--model_type', type=str, choices=['gabmil', 'transmil'], default='gabmil', help='type of model (default: gabmil)')

args = parser.parse_args()

# Model settings
print('\nInit Model...', end=' ')
model_dict = {"use_local": args.use_local, "use_block": args.use_block, "use_grid": args.use_grid, "win_size_b": args.win_size_b, "win_size_g": args.win_size_g}

if args.model_type == 'transmil':
    model = TransMIL()
else:    
    model = GABMIL(**model_dict)
model.relocate()
print('Done!')

print_network(model)

input_shape = (120, 1024)
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=2)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
