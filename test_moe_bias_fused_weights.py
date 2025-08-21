import torch
import pytest
from typing import Optional
import os

# Set environment variables before importing vLLM modules
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

from vllm.model_executor.layers.fused_moe.layer import FusedMoE, UnquantizedFusedMoEMethod
from vllm.platforms import current_platform
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    destroy_model_parallel,
    destroy_distributed_environment
)

def setup_distributed_environment():
    """Initialize the distributed environment for testing"""
    try:
        # Initialize distributed environment
        if not torch.distributed.is_initialized():
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )
        
        # Initialize model parallel groups
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1
        )
        print("✓ Distributed environment initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize distributed environment: {e}")
        print("This might be expected in some test environments")

def cleanup_distributed_environment():
    """Clean up the distributed environment"""
    try:
        destroy_model_parallel()
        destroy_distributed_environment()
        print("✓ Distributed environment cleaned up")
    except Exception as e:
        print(f"Warning: Could not clean up distributed environment: {e}")

def test_hpu_vs_native_moe_implementation():
    """
    Unit test to compare HPU-optimized MoE implementation with PyTorch native implementation.
    Tests both forward pass outputs and ensures numerical equivalence.
    """
    
    # Initialize distributed environment
    setup_distributed_environment()
    
    try:
        # Test configuration
        batch_size = 4
        seq_len = 64
        hidden_size = 2880
        intermediate_size = 2880
        num_experts = 32
        top_k = 4
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create test data
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
        router_logits = torch.randn(batch_size, seq_len, num_experts, dtype=torch.bfloat16)
        
        if current_platform.is_hpu():
            hidden_states = hidden_states.to('hpu')
            router_logits = router_logits.to('hpu')
        
        # Create MoE layer with explicit parameters
        moe_layer = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=torch.bfloat16,
            activation="oss_act",  # Use OSS activation for GPT-OSS
            tp_size=1,  # Explicitly set tensor parallel size
            ep_size=None,  # Let it default
            dp_size=None,  # Let it default
            prefix="test_moe"  # Add a prefix for identification
        )
        
        if current_platform.is_hpu():
            moe_layer = moe_layer.to('hpu')
        
        # Initialize weights with known values for reproducible testing
        torch.manual_seed(42)
        with torch.no_grad():
            # Initialize w13_weight (gate_up combined)
            moe_layer.w13_weight.normal_(0, 0.02)
            # Initialize w2_weight (down projection)
            moe_layer.w2_weight.normal_(0, 0.02)
            # Initialize biases
            moe_layer.w13_bias.zero_()
            moe_layer.w2_bias.zero_()
        
        # Process weights after loading - this is crucial for HPU implementation
        # This sets up the HPU-specific MoeMatmul weights and biases
        moe_layer.quant_method.process_weights_after_loading(moe_layer)
        
        # Test 1: Native PyTorch implementation (using fused_moe function)
        print("Testing native PyTorch implementation...")
        
        # Temporarily enable native implementation by calling fused_moe directly
        quant_method = moe_layer.quant_method
        assert isinstance(quant_method, UnquantizedFusedMoEMethod)
        
        # Call the fused_moe function directly (native implementation)
        with torch.no_grad():
            native_output = quant_method.fused_moe(
                hidden_states=hidden_states,
                w1=moe_layer.w13_weight,
                w2=moe_layer.w2_weight,
                w1_bias=moe_layer.w13_bias,
                w2_bias=moe_layer.w2_bias,
                gating_output=router_logits,
                topk=top_k,
                global_num_experts=num_experts,
                renormalize=True
            )

        # Test 2: HPU-optimized implementation
        print("Testing HPU-optimized implementation...")
        
        if current_platform.is_hpu():
            # Call the HPU forward implementation
            with torch.no_grad():
                hpu_output = quant_method.forward_hpu(
                    layer=moe_layer,
                    x=hidden_states,
                    use_grouped_topk=False,
                    top_k=top_k,
                    router_logits=router_logits,
                    renormalize=True,
                    global_num_experts=num_experts,
                    activation="oss_act"
                )

        # Test 3: Compare outputs
        print("Comparing outputs...")
        
        if current_platform.is_hpu():
            # Convert to CPU for comparison if needed
            native_output_cpu = native_output.cpu() if hasattr(native_output, 'cpu') else native_output
            hpu_output_cpu = hpu_output.cpu() if hasattr(hpu_output, 'cpu') else hpu_output
            
            # Check shapes match
            assert native_output_cpu.shape == hpu_output_cpu.shape, \
                f"Shape mismatch: native {native_output_cpu.shape} vs HPU {hpu_output_cpu.shape}"
            
            # out = native_output_cpu==hpu_output_cpu
            # print(out)
            
            
            # Check numerical equivalence with tolerance for floating point differences
            max_diff = torch.max(torch.abs(native_output_cpu - hpu_output_cpu)).item()
            mean_diff = torch.mean(torch.abs(native_output_cpu - hpu_output_cpu)).item()
            
            print(f"Max absolute difference: {max_diff}")
            print(f"Mean absolute difference: {mean_diff}")
            
            # Assert numerical equivalence (adjust tolerance as needed)
            # assert max_diff < 1e-2, f"Max difference {max_diff} exceeds tolerance"
            torch.testing.assert_close(native_output_cpu, hpu_output_cpu, atol=0.01, rtol=0.01)
            
            
            # Check for NaN or Inf values
            assert not torch.isnan(native_output_cpu).any(), "Native output contains NaN"
            assert not torch.isnan(hpu_output_cpu).any(), "HPU output contains NaN"
            assert not torch.isinf(native_output_cpu).any(), "Native output contains Inf"
            assert not torch.isinf(hpu_output_cpu).any(), "HPU output contains Inf"
        
            
            print("✓ HPU and native implementations produce equivalent results!")
        else:
            print("✓ Test completed (HPU comparison skipped on non-HPU device)")
    
    finally:
        # Always clean up distributed environment
        cleanup_distributed_environment()


if __name__ == "__main__":
    print("Running MoE HPU vs Native Implementation Tests")
    print("=" * 60)
    
    try:
        # Run the main comparison test
        test_hpu_vs_native_moe_implementation()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to clean up anyway
        try:
            cleanup_distributed_environment()
        except:
            pass
        
        raise
