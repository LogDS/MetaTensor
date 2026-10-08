import torch

# 1. Load the compiled LibTorch JIT module
model_path = "create_user_loss_and_stopping_subgraph.pt"
try:
    module = torch.jit.load(model_path)
    print(f" Successfully loaded: {model_path}\n")
except Exception as e:
    print(f"Failed to load module: {e}")
    exit(1)

# 2. Inspect the function signature to verify the schema we defined in C++
print("--- Method Schema ---")
print(module.forward.schema)
print("---------------------\n")

# 3. Prepare mock data matching the graph's expected types
# Arguments layout: (weights, inputs_list, targets_list, loop_iter, loop_cond_in)

# weights: List[Tensor] (we fetch index 0 inside the graph, shape chosen: 4x1)
weights = [torch.randn(4, 1)]

# inputs_list: List[Tensor] containing 2 batches (since num_batches = 2 in C++)
inputs_list = [
    torch.randn(3, 4),  # Batch 0 (3 samples, 4 features)
    torch.randn(3, 4)   # Batch 1 (3 samples, 4 features)
]

# targets_list: List[Tensor] containing target shapes matching the forward matmul output (3, 1)
targets_list = [
    torch.randn(3, 1),  # Targets for Batch 0
    torch.randn(3, 1)   # Targets for Batch 1
]

# Primitive scalars for loop execution control
loop_iter = 0          # Will resolve to index 0 via (0 % 2) remainder node
loop_cond_in = True    # Initial tracking condition

# 4. Run the module container forward pass
print("--- Executing Iteration 0 ---")
loss, continue_loop = module(weights, inputs_list, targets_list, loop_iter, loop_cond_in)

print(f"Output Loss: {loss.item():.6f}")
print(f"Continue Loop Signal: {continue_loop}")

# 5. Run it again with a different iteration index to verify dynamic batch tracking
print("\n--- Executing Iteration 1 ---")
loop_iter_next = 1     # Will resolve to index 1 via (1 % 2) remainder node
loss_next, continue_loop_next = module(weights, inputs_list, targets_list, loop_iter_next, loop_cond_in)

print(f"Output Loss (Batch 1): {loss_next.item():.6f}")
print(f"Continue Loop Signal: {continue_loop_next}")
