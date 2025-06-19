'''
consider an image of size 308 x 364

Aligned to 112, it will be: 336 x 448

If we divide by 14:
24 x 32

Lets say our nearest bucket is 1120x1120
which is: 80x80 = 6400


So we have orig img: 308 x 364 -> 22 x 26 = 572
and nearest bucket is 6400



| 8,8 | 8,8 | 8,6 |
| 8,8 | 8,8 | 8,6 |
| 8,8 | 8,8 | 8,6 |
| 2,8 | 2,8 | 2,6 |
Fig 1





This is the basic blocking... inner blocks are full 8x8
side blocks may be full-row semi column
the bottom right block may be semi-row-semi-col


Now we also need to pad this to 6400, those will all have empty/zero masks



grid_thw
[22, 26]


and we have a for loop that iterates 6400/64 = 100 times and slices qkv



Lets name the cells
| 0 | 1 | 2 |
| 3 | 4 | 5 |
| 6 | 7 | 8 |
| 9 | 10 | 11 |
Fig 2


| 0,0 | 0,1 | 0,2 |
| 1,0 | 1,1 | 1,2 |
| 2,0 | 2,1 | 2,2 |
| 3,0 | 3,1 | 3,2 |
Fig 3


'''

import numpy as np
from PIL import Image

def mask_maker(block_row_id, block_col_id, num_rows_of_blocks, num_cols_of_blocks, right_edge_cols, bottom_edge_rows):
    '''
    For full mask:
    x = [1,1,1,1,1,1,1,1]
    for empty mask:
    x = [0,0,0,0,0,0,0,0]
    for full-row, semi-colm (right edge):
    ...?
    for full col, semi-row (bottom edge):
    ...?
    for bottom right corner:
    x = [1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,]
    '''

    # if you are one of the inner (non-border) rows, you are guaranteed to have all 8 rows.
    # vice versa for inner (non-border) cols
    full_row = block_row_id < num_rows_of_blocks-1
    full_col = block_col_id < num_cols_of_blocks-1

    #breakpoint()

    # right now using python if-else, use torch.where later
    num_rows = torch.where(full_row, 8, bottom_edge_rows)
    num_cols = torch.where(full_col, 8, right_edge_cols)

    tmp = torch.arange(8) < num_cols
    single_8x8 = torch.outer(tmp,tmp)
    mask_64x64 = single_8x8.repeat((8,8))

    # now wipe out last few rows
    mask_blackout_rows = (torch.arange(64*64).view((64,64)) >= (64*8*(num_rows))) # if bottom_edge_rows is 2, we need to zero out the last 2*8=16 rows
    mask_blackout_cols = mask_blackout_rows.t()
    #not_full_col_mask = torch.logical_or(mask_blackout_rows, mask_blackout_cols)
    # only when not full_rows, apply torch.logical_or(mask_blackout_rows, mask_blackout_cols)


    final_mask = torch.where(torch.logical_or(mask_blackout_rows, mask_blackout_cols), 0, mask_64x64)

    Image.fromarray((final_mask==1).cpu().squeeze().numpy().astype(np.uint8)*255).save(f'{num_rows_of_blocks}_{num_cols_of_blocks}_{right_edge_cols}_{bottom_edge_rows}/{block_row_id}_{block_col_id}.png')
    
    if full_row.item() and full_col.item(): # inner blocks
        assert final_mask.sum() == 64*64
    if not full_row.item() and full_col.item(): # bottom edge
        assert final_mask.sum() == num_rows*8*num_rows*8
    if full_row.item() and not full_col.item(): # bottom edge
        assert final_mask.sum() == num_cols*8*num_cols*8
    if not full_row.item() and not full_col.item():
        assert final_mask.sum() == num_cols*num_cols*num_rows*num_rows
    

import torch
import math, os
import shutil
bucket_size = 1600 # a small number of fixed buckets
block_size = 64 # fixed
num_cols_img = 308 / 14 # = 22
num_rows_img = 364 / 14 # = 26
right_edge_cols = num_cols_img % 8 # = 6
bottom_edge_rows = num_rows_img % 8 # = 2
num_cols_of_blocks = math.ceil(num_cols_img / 8) # = 3
num_rows_of_blocks = math.ceil(num_rows_img / 8) # = 4

#mask_maker(torch.tensor(0), torch.tensor(2), num_rows_of_blocks, num_cols_of_blocks, right_edge_cols, bottom_edge_rows)

directory_path = f'{num_rows_of_blocks}_{num_cols_of_blocks}_{right_edge_cols}_{bottom_edge_rows}'
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    shutil.rmtree(directory_path)
os.mkdir(directory_path)

for i in torch.arange(bucket_size/block_size):
    # from i, we need a unique mapping to the blocks shown in Fig 1
    # See Fig 2 and Fig 3 for 1D and 2D index
    block_row_id = i // num_cols_of_blocks
    block_col_id = i % num_cols_of_blocks

    attn_mask = mask_maker(block_row_id, block_col_id, num_rows_of_blocks, num_cols_of_blocks, right_edge_cols, bottom_edge_rows)
