import torch

# 1. Load the compiled LibTorch JIT module
model_path = "/home/gyankos/CLionProjects/TensorLibrary/cmake-build-debug-gcc12/create_parametrizable_optimizer_subgraph.pt"
print(f"Loading TorchScript module from: {model_path}")
module = torch.jit.load(model_path)

# 2. Setup dummy testing data matching your 3-tensor configuration
# We create 3 independent tensors with arbitrary tracking shapes
shapes = [(2, 2), (3, 1), (5,)]

weights = [torch.ones(shape, dtype=torch.float32) for shape in shapes]
grads = [torch.full(shape, 0.1, dtype=torch.float32) for shape in shapes]
m_in = [torch.zeros(shape, dtype=torch.float32) for shape in shapes]
v_in = [torch.zeros(shape, dtype=torch.float32) for shape in shapes]

# Scalar primitives matching your TorchScript arguments schema
lr_in = 0.01
loop_iter = 1  # Used for testing the Exponential/Step decay rules

print("\n--- Input Parameters ---")
print(f"Initial learning rate: {lr_in}")
print(f"Initial weights sample (first tensor top row): {weights[0][0].tolist()}")

# 3. Execute the module via its bound forward method
# Signature mapping: forward(weights_in, grads_in, lr_in, m_in, v_in, loop_iter)
try:
    outputs = module(weights, grads, lr_in, m_in, v_in, loop_iter)

    # 4. Unpack the returned TorchScript tuple structure
    next_weights, next_lr, next_m, next_v = outputs

    print("\n--- Output Validation ---")
    print(f"New updated Learning Rate: {next_lr:.6f}")

    print("\nWeights checking:")
    for i, (w_old, w_new) in enumerate(zip(weights, next_weights)):
        print(f"  Tensor {i} shape {tuple(w_old.shape)} | All weights modified: {not torch.equal(w_old, w_new)}")

    print("\nState Buffer validation:")
    print(f"  Adam M1 buffer initialized and modified: {torch.any(next_m[0] != 0).item()}")
    print(f"  Adam M2 variance buffer modified:        {torch.any(next_v[0] != 0).item()}")

    # Simple mathematical verification assertion for Adam optimization
    # Weights should step in the opposite direction of the gradient sign
    assert next_weights[0][0, 0] < weights[0][0, 0], "Weight step logic anomaly detected!"
    print("\n[SUCCESS] Script executed and unpacked safely.")

except Exception as e:
    print(f"\n[FAILURE] Execution crash: {e}")
