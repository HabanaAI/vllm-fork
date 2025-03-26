import torch
import torch.nn.functional as F


class TestClass:
    def __init__(self):
        self.window_size = 112
        self.spatial_merge_size = 2
        self.patch_size = 14
        self.spatial_merge_unit = self.spatial_merge_size**2

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )
        print(vit_merger_window_size)
        # import pdb;pdb.set_trace()
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            # TODO: This arange, reshape based on the value of tensor can't work in WARMUP, or with TENSOR_CACHE
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            #print(pad_h, pad_w)
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            print(num_windows_h ,num_windows_w)
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            print(index_padded.size())
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            print(seqlens.size())
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens


# Create an instance of the class
test_instance = TestClass()

# Create a tensor for testing
grid_thw = torch.tensor([[1, 7, 86]])
#grid_thw = torch.tensor([[1, 72, 72], [1, 2, 608]])

# Run the abc method
window_index, cu_window_seqlens = test_instance.get_window_index(grid_thw)
cu_window_seqlens_delta = (
    torch.tensor(cu_window_seqlens)[1:] - torch.tensor(cu_window_seqlens)[:-1]
)
for i in ["window_index", "cu_window_seqlens", "cu_window_seqlens_delta"]:
    print(f"var {i}: {globals()[i]}")
