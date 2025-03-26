import torch
import torch.nn.functional as F
import argparse

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


def make_multiple_of_28(n):
    return ((n + 27) // 28) * 28

def generate_image(h,w):
    from vllm.assets.image import ImageAsset
    import PIL
    from transformers import AutoProcessor  # noqa: F401
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # image to pixel values
    image_processor = processor.image_processor

    image = ImageAsset("stop_sign").pil_image
    image = image.resize((w, h))

    preprocess_result = image_processor \
        .preprocess(images=image, return_tensors="pt") \
        .data
    pixel_values = preprocess_result["pixel_values"]
    image_grid_thw = preprocess_result["image_grid_thw"]

    print("pixel_values :", {pixel_values.shape})
    print("image_grid_thw :", {image_grid_thw})

    return pixel_values, image_grid_thw


#def main():
parser = argparse.ArgumentParser(description="Process get_window_index from image")
parser.add_argument('--width', type=int, required=True, help='Width of the image')
parser.add_argument('--height', type=int, required=True, help='Height of the image')

args = parser.parse_args()

print(f"Image Height: {args.height}, Width: {args.width},")

# Create an instance of the class
test_instance = TestClass()

w=args.width
h=args.height

pixel_values, grid_thw = generate_image(h,w)


# Run the abc method
window_index, cu_window_seqlens = test_instance.get_window_index(grid_thw)
cu_window_seqlens_delta = (
    torch.tensor(cu_window_seqlens)[1:] - torch.tensor(cu_window_seqlens)[:-1]
)

for i in ["window_index", "cu_window_seqlens", "cu_window_seqlens_delta"]:
    print(f"var {i}: {globals()[i]}")
