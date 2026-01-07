import requests
import base64
import io
import numpy as np
import argparse


def send_tensor_to_server(pixel_values: np.ndarray, server_url: str):
    # Prepare request
    buffer = io.BytesIO()
    np.save(buffer, pixel_values)
    buffer.seek(0)
    tensor_bytes = buffer.read()
    tensor_b64 = base64.b64encode(tensor_bytes).decode('utf-8')
    
    payload = {
        "tensor": tensor_b64,
        "shape": list(pixel_values.shape),
        "dtype": str(pixel_values.dtype)
    }
    
    print(f"+-+ Sending tensor to {server_url}/encode")
    
    # Send to server
    response = requests.post(f"{server_url}/encode", json=payload)
    
    if response.status_code != 200:
        print(f"+-+ Error: {response.status_code}")
        print(f"+-+ Response: {response.text}")
        return None

    result = response.json()
    print(f"+-+ Success! Received embeddings with shapes: {result['shape']}")

    # Convert to tensor
    embeddings_bytes = base64.b64decode(result['embeddings'])
    embeddings_buffer = io.BytesIO(embeddings_bytes)
    embeddings = np.load(embeddings_buffer)
    
    return embeddings


def compare_tensors(t1, t2):
    t1_f64 = t1.astype(np.float64)
    t2_f64 = t2.astype(np.float64)

    print(f"\n+-+ Comparing tensors...")
    print(f"+-+ t1 shape={t1.shape} dtype={t1.dtype} device={t1.device}")
    print(f"+-+ t2 shape={t2.shape} dtype={t2.dtype} device={t2.device}")

    # MSE
    mse = np.mean((t1_f64 - t2_f64) ** 2)

    # Cosine Distance
    t1_f64_flat = t1_f64.flatten()
    t2_f64_flat = t2_f64.flatten()
    cosine_sim = np.dot(t1_f64_flat, t2_f64_flat) / (np.linalg.norm(t1_f64_flat) * np.linalg.norm(t2_f64_flat))
    cosine_dist = 1 - cosine_sim
   
    print(f"+-+ MSE: {mse:{5}.{3}f}")
    print(f"+-+ Cosine Distance: {cosine_dist:{5}.{3}f}")
    return 


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="VT debug client")
    parser.add_argument("--pixel-values", required=True, help="Path to input pixel that we got from gemma (.npy file)")
    parser.add_argument("--ref-embedding", required=True, help="Path to reference embedding that we got from gemma (.npy file)")
    parser.add_argument("--out-embedding", required=True, help="Path to computed embedding that we are going to send to gemma (.npy file)")
    parser.add_argument("--server-url", default="http://localhost:8000", help="Server URL")
    args = parser.parse_args()
    
    # Load pixel tensor
    pixel_values = np.load(args.pixel_values)
    print(f"+-+ pixel_values from={args.pixel_values}")
    print(f"+-+ pixel_values shape={pixel_values.shape} dtype={pixel_values.dtype} device={pixel_values.device}")

    # Load ref_ten tensor
    ref_embeddings = np.load(args.ref_embedding)
    print(f"+-+ ref_embeddings from={args.ref_embedding}")
    print(f"+-+ ref_embeddings shape={ref_embeddings.shape} dtype={ref_embeddings.dtype} device={ref_embeddings.device}")

    # Send pixel tensor to server
    out_embeddings = send_tensor_to_server(pixel_values, args.server_url)

    # save computed embeddings
    np.save(args.out_embedding, out_embeddings)
    print(f"+-+ out_embeddings shape={out_embeddings.shape} dtype={out_embeddings.dtype} device={out_embeddings.device}")
    print(f"+-+ out_embeddings saved to {args.out_embedding}")

    # compare results
    if out_embeddings is not None:
        compare_tensors(out_embeddings, ref_embeddings)
        print(f"+-+ Done!")
