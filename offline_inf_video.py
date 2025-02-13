# Process video input and select specified number of evenly distributed frames
def sample_frames(path, num_frames):
    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    print("Total frames" , total_frames)
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            pil_img = pil_img.resize((256, 256))
            frames.append(pil_img)
    video.release()
    return frames[:num_frames]

# Video Input
video = sample_frames("/tmp/cats_1.mp4", 50)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95 )
llm = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct", enforce_eager=True, dtype='bfloat16' , gpu_memory_utilization=0.4)

prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>what is the cat doing.<|im_end|>\n<|im_start|>assistant\n'
outputs = llm.generate({"prompt": prompt, "multi_modal_data": {"video": video}})

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
