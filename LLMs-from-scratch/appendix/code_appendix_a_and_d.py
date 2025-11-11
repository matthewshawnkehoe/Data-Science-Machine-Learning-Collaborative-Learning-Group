import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt

"""
====================================================

Based on "Build a Large Language Model From Scratch" by Sebastian Raschka
Repository: https://github.com/rasbt/LLMs-from-scratch

This script covers:
- A.1 What is PyTorch
- A.2 Understanding tensors
- A.3 Seeing models as computation graphs
- A.4 Automatic differentiation
- A.5 Implementing multilayer neural networks
- A.6 Setting up efficient data loaders
- A.7 A typical training loop
- A.8 Saving and loading models
- D.1 Additional Techniques
"""


def check_pytorch_setup():
    """Check PyTorch installation and GPU availability."""
    print()
    print("Are PyTorch and CUDA available?")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())


def create_tensors():
    """
    Creating tensors of different dimensions. Tensors are the fundamental data structure in PyTorch.
    """
    print("Creating Tensors\n")

    # TODO: Create a 0D tensor (scalar) from a Python integer
    tensor0d = None

    # TODO: Create a 1D tensor (vector) from a Python list
    tensor1d = None

    # TODO: Create a 2D tensor (matrix) from a nested Python list
    tensor2d = None

    # TODO: Create a 3D tensor from a nested Python list
    # Example: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    tensor3d = None

    if tensor0d is not None:
        print(f"0D tensor (scalar): {tensor0d}")
        print(f"Shape: {tensor0d.shape}\n")

    if tensor1d is not None:
        print(f"1D tensor (vector): {tensor1d}")
        print(f"Shape: {tensor1d.shape}\n")

    if tensor2d is not None:
        print(f"2D tensor (matrix):\n{tensor2d}")
        print(f"Shape: {tensor2d.shape}\n")

    if tensor3d is not None:
        print(f"3D tensor:\n{tensor3d}")
        print(f"Shape: {tensor3d.shape}\n")


def numpy_tensor_conversion():
    """
    Convert between NumPy arrays and PyTorch tensors. The major difference is
    that torch.tensor() copies data while torch.from_numpy() shares memory.
    """
    print("NumPy and PyTorch Conversion\n")

    # NumPy array
    ary3d = np.array([[[1, 2], [3, 4]],
                      [[5, 6], [7, 8]]])

    # TODO: Convert NumPy array to tensor (create a copy)
    tensor3d_copy = None

    # TODO: Convert NumPy array to tensor (shares memory)
    tensor3d_shared = None  # torch.from_numpy()

    # Modify the NumPy array
    ary3d[0, 1, 0] = 99

    print("After modifying NumPy array to 999 at position [0,1,0]:")
    if tensor3d_copy is not None:
        print(f"Copied tensor (remains unchanged):\n{tensor3d_copy}\n")
    if tensor3d_shared is not None:
        print(f"Shared memory tensor (changes with NumPy):\n{tensor3d_shared}\n")


def tensor_data_types():
    """
    Explore tensor data types and type conversion.
    Common types: torch.float32, torch.int64, torch.float64
    """
    print("Tensor Data Types\n")

    # TODO: Create an integer tensor and check its data type
    int_tensor = None  # Try [1, 2, 3]

    if int_tensor is not None:
        print(f"Integer tensor: {int_tensor}")
        print(f"Data type: {int_tensor.dtype}\n")

    # TODO: Create a float tensor
    float_tensor = None  # Try [0.99999999, 1.99999999, 2.99999999]

    if float_tensor is not None:
        print(f"Float tensor: {float_tensor}")
        print(f"Data type: {float_tensor.dtype}\n")

    # TODO: Convert integer tensor to float32
    # Hint: Use .to(torch.float32)
    converted_tensor = None

    if converted_tensor is not None:
        print(f"Converted to float32: {converted_tensor}")
        print(f"Data type: {converted_tensor.dtype}\n")


def tensor_operations():
    """
    PyTorch tensor operations include shape,  reshape, view, transpose, and matrix multiplication.
    """
    print("Tensor Operations\n")

    # Create a sample 2D tensor
    tensor2d = torch.tensor([[1, 2, 3],
                             [4, 5, 6]])
    print(f"Original tensor:\n{tensor2d}")
    print(f"Shape: {tensor2d.shape}\n")

    # TODO: Reshape the tensor to (3, 2)
    reshaped = None

    if reshaped is not None:
        print(f"Reshaped to (3, 2):\n{reshaped}\n")

    # TODO: Use view to reshape (similar as reshape but saves memory)
    viewed = None

    if viewed is not None:
        print(f"Viewed as (3, 2):\n{viewed}\n")

    # TODO: Transpose the tensor
    transposed = None

    if transposed is not None:
        print(f"Transposed:\n{transposed}\n")

    # TODO: Matrix multiplication with its transpose
    # This should give a 2x2 result
    matmul_result = None

    if matmul_result is not None:
        print(f"Matrix multiplication (tensor2d @ tensor2d.T):\n{matmul_result}\n")


def computation_graph():
    """
    Computation graph for a single neuron which shows how neural networks perform forward computations.
    """
    print("Computation Graph (Single Neuron)\n")

    # TODO: Create the following tensors:
    y = None  # True label: [1.0]
    x1 = None  # Input feature: [1.1]
    w1 = None  # Weight parameter: [2.2]
    b = None  # Bias unit: [0.0]

    # TODO: Compute net input: z = x1 * w1 + b
    z = None

    # TODO: Apply sigmoid activation
    a = None

    # TODO: Compute binary cross entropy loss through F.binary_cross_entropy
    loss = None

    if loss is not None:
        print(f"Input: {x1}")
        print(f"Weight: {w1}")
        print(f"Bias: {b}")
        print(f"Net input (z): {z}")
        print(f"Activation (a): {a}")
        print(f"True label: {y}")
        print(f"Loss: {loss}\n")


def automatic_differentiation():
    """
    Automatic differentiation is doen in PyTorch through autograd.
    """
    print("Automatic Differentiation\n")

    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])

    # TODO: Create tensors with requires_grad=True to track gradients
    w1 = None  # Use torch.tensor with requires_grad=True
    b = None

    # TODO: Forward pass
    z = None  # x1 * w1 + b
    a = None  # torch.sigmoid([])
    loss = None  # F.binary_cross_entropy

    # TODO: Compute gradients using backward()

    if w1 is not None and b is not None:
        print(f"Loss: {loss}")
        print(f"Gradient w.r.t. w1: {w1.grad}")
        print(f"Gradient w.r.t. b: {b.grad}\n")


class NeuralNetwork(torch.nn.Module):
    """
    A feedforward neural network with:
    - Input layer
    - Two hidden layers (30 and 20 neurons)
    - Output layer
    """

    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        # TODO: Define the layers using torch.nn.Sequential
        # Layer 1: Linear(num_inputs, 30) -> ReLU
        # Layer 2: Linear(30, 20) -> ReLU
        # Layer 3: Linear(20, num_outputs)

        self.layers = None # torch.nn.Sequential ...

    def forward(self, x):
        """Forward pass through the network."""
        # TODO: Pass input through the layers
        logits = None
        return logits


def create_and_inspect_model():
    """
    Create a neural network and inspect its structure.
    """
    print("Creating and Inspecting Neural Network\n")

    torch.manual_seed(123)

    # TODO: Create a model with 50 inputs and 3 outputs
    model = None

    if model is not None:
        print(f"Model architecture:\n{model}\n")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {num_params}\n")

        X = torch.rand((1,50))

        if X is not None:
            # TODO: Get model output
            out = None  # Use model(something)

            if out is not None:
                print(f"Model output (logits): {out}\n")

                # TODO: Apply softmax to get probabilities
                # Use torch.softmax(out, dim=1) with torch.no_grad()
                probabilities = None

                if probabilities is not None:
                    print(f"Output probabilities: {probabilities}\n")


def create_toy_dataset():
    """
    Create a toy dataset for binary classification.
    """
    from torch.utils.data import Dataset

    class ToyDataset(Dataset):
        """A toy dataset class."""

        def __init__(self, X, y):
            self.features = X
            self.labels = y

        def __getitem__(self, index):
            """Get a single item from the dataset."""
            # TODO: Return the features and label at the given index
            one_x = None
            one_y = None
            return one_x, one_y

        def __len__(self):
            """Return the size of the dataset."""
            # TODO: Return the number of samples
            return 0  # Replace with actual length

    return ToyDataset


def setup_data_loaders():
    """
    Create data loaders for training and testing. DataLoaders handle batching, shuffling, and loading.
    """

    print("Setting Up Data Loaders\n")

    # Create training data
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    # Create test data
    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # TODO: Create dataset instances
    ToyDataset = create_toy_dataset()
    train_ds = None  # Instantiate ToyDataset
    test_ds = None

    # TODO: Create data loaders
    # Use DataLoader with batch_size=2, shuffle=True for training
    torch.manual_seed(123)
    train_loader = None
    test_loader = None

    if train_loader is not None:
        print(f"Number of training samples: {len(train_ds) if train_ds else 0}")
        print(f"Number of test samples: {len(test_ds) if test_ds else 0}\n")

        print("Training batches:")
        for idx, (x, y) in enumerate(train_loader):
            print(f"  Batch {idx + 1}: Features shape {x.shape}, Labels shape {y.shape}")

    return train_loader, test_loader


def train_model():
    """
    Implement a training loop in PyTorch.
    """
    print("Train the Model\n")

    torch.manual_seed(123)
    train_loader, test_loader = setup_data_loaders()

    if train_loader is None:
        print("Complete data loaders before training")
        return

    # TODO: Create the model (2 inputs, 2 outputs for binary classification)
    model = None

    # TODO: Create optimizer (Adam with learning rate 0.5)
    optimizer = None

    num_epochs = 3

    if model is not None and optimizer is not None:
        for epoch in range(num_epochs):
            # Set model to training mode
            model.train()

            for batch_idx, (features, labels) in enumerate(train_loader):
                # TODO: Forward pass - get model predictions (logits)
                logits = None  # What are the model predictions?

                # TODO: Compute loss using cross entropy
                # Use F.cross_entropy loss on [] and []?
                loss = None

                # TODO: Backward pass and optimization
                # Step 1: Zero the gradients
                # Step 2: Compute gradients
                # Step 3: Update weights


                if loss is not None:
                    print(f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                          f" | Batch {batch_idx + 1:02d}/{len(train_loader):02d}"
                          f" | Loss: {loss:.4f}")

        return model
    else:
        print("Please implement model and optimizer creation first!")
        return None


def compute_accuracy(model, dataloader):
    """
    Compute accuracy of the model on a dataset.

    Args:
        model: The neural network model
        dataloader: DataLoader containing the dataset

    Returns:
        Accuracy as a float between 0 and 1
    """
    if model is None or dataloader is None:
        return 0.0

    model.eval()  # Set to evaluation mode
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        # TODO: Get model predictions without computing gradients
        with torch.no_grad():
            logits = None  # Same as in train(model)

        # TODO: Get predicted class (argmax of logits)
        predictions = None  # Use torch.argmax across what dimension?

        # TODO: Count correct predictions
        if predictions is not None:
            compare = None  # Correct is when labels is equal to predictions
            correct += torch.sum(compare)
            total_examples += len(compare)

    return (correct / total_examples).item() if total_examples > 0 else 0.0


def evaluate_model():
    """Evaluate the trained model."""
    print("Evaluating the Model\n")

    # Train the model
    model = train_model()

    if model is not None:
        train_loader, test_loader = setup_data_loaders()

        # TODO: Compute training and test accuracy
        train_acc = None  # Use the function defined above and the appropriate data loader
        test_acc = None # Same comment as previous

        if train_acc is not None and test_acc is not None:
            print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
            print(f"Test Accuracy: {test_acc * 100:.2f}%")

        return model


def save_and_load_model():
    """
    Demonstrate saving and loading model weights.
    """
    print("Saving and Loading Models\n")

    # Train a model
    model = train_model()

    if model is not None:
        # TODO: Save the model's state dictionary
        # Use torch.save on the model's state dictionary to "model.pth"

        print("Model saved to 'model.pth'\n")

        # TODO: Create a new model instance
        new_model = None  # Instantiate the NeuralNetwork class with 2 inputs and 2 ouputs

        # TODO: Load the saved weights
        # Use new_model.load_state_dict in combiantion with torch.load and weights_only=True

        if new_model is not None:
            print("Model loaded successfully!")

            # Verify it works
            train_loader, _ = setup_data_loaders()
            if train_loader is not None:
                acc = compute_accuracy(new_model, train_loader)
                print(f"Loaded model accuracy: {acc * 100:.2f}%")


def learning_rate_warmup():
    """
    Learning rate warmup. Warmup gradually increases learning rate from initial_lr to peak_lr.
    This helps stabilize training by starting with small weight updates.
    """
    print("Learning Rate Warmup\n")

    # Training parameters
    n_epochs = 15
    initial_lr = 0.0001
    peak_lr = 0.01

    # Create dummy data loader
    train_loader, _ = setup_data_loaders()
    if train_loader is None:
        print("Implement data loaders first.")
        return

    total_steps = len(train_loader) * n_epochs

    # warmup steps (20% of total)
    warmup_steps = int(0.2 * total_steps)

    print(f"Total training steps: {total_steps if total_steps else '?'}")
    print(f"Warmup steps: {warmup_steps if warmup_steps else '?'}\n")

    if total_steps is not None and warmup_steps is not None:
        lr_increment = (peak_lr - initial_lr) / warmup_steps

        global_step = -1
        track_lrs = []

        # Create a model and optimizer
        torch.manual_seed(123)
        model = NeuralNetwork(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)

        # Simulate training loop
        for epoch in range(n_epochs):
            for input_batch, target_batch in train_loader:
                global_step += 1

                # TODO: Calculate current learning rate
                # If we're in warmup phase: lr = initial_lr + global_step * lr_increment
                # Otherwise: lr = peak_lr
                if global_step < warmup_steps:
                    lr = initial_lr + global_step * lr_increment
                else:
                    lr = peak_lr

                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                track_lrs.append(optimizer.param_groups[0]["lr"])

                if lr is not None:
                    track_lrs.append(lr)

        # Plot the learning rate schedule
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(track_lrs)), track_lrs)
        plt.ylabel("Learning rate")
        plt.xlabel("Step")
        plt.title("Learning Rate Warmup Schedule")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def cosine_annealing():
    """
    Cosine annealing (cosine decay) with warmup. After warmup, learning rate
    follows a cosine curve from peak_lr to min_lr. This provides smoother learning rate
    reduction than linear decay.
    """

    print("Cosine Annealing with Warmup\n")

    # Training parameters
    n_epochs = 15
    initial_lr = 0.0001
    peak_lr = 0.01
    min_lr = 0.1 * initial_lr

    train_loader, _ = setup_data_loaders()
    if train_loader is None:
        print("Implement data loaders first.")
        return

    # Calculate number of steps
    total_steps = len(train_loader) * n_epochs
    warmup_steps = int(0.2 * total_steps)

    print(f"Peak learning rate: {peak_lr}")
    print(f"Minimum learning rate: {min_lr}\n")

    lr_increment = (peak_lr - initial_lr) / warmup_steps

    global_step = -1
    track_lrs = []

    torch.manual_seed(123)
    model = NeuralNetwork(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)

    for epoch in range(n_epochs):
        for input_batch, target_batch in train_loader:
            global_step += 1

            # Calculate learning rate with warmup and cosine annealing
            if global_step < warmup_steps:
                # Linear warmup
                lr = initial_lr + global_step * lr_increment
            else:
                # Cosine annealing after warmup
                progress = ((global_step - warmup_steps) / (total_steps - warmup_steps))

                # Apply cosine formula: min_lr + (peak_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # Apply learning rate to optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)

    # Plot the learning rate schedule
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(track_lrs)), track_lrs)
    plt.ylabel("Learning rate")
    plt.xlabel("Step")
    plt.title("Cosine Annealing with Warmup")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def gradient_clipping():
    """
    Demonstrate gradient clipping to prevent exploding gradients.
    Clips gradient norm to a maximum value (e.g., 1.0).

    The L2 norm of gradients is: ||G||_2 = sqrt(sum of all squared gradient values)
    """
    print("Gradient Clipping\n")

    # Helper function to find the highest gradient
    def find_highest_gradient(model):
        """Find the maximum gradient value across all model parameters."""
        max_grad = None
        for param in model.parameters():
            if param.grad is not None:
                grad_values = param.grad.data.flatten()
                max_grad_param = grad_values.max()

                if max_grad is None or (max_grad_param is not None and max_grad_param > max_grad):
                    max_grad = max_grad_param
        return max_grad

    # Create a model and compute gradients
    torch.manual_seed(123)
    model = NeuralNetwork(2, 2)

    # Obtain a batch of data
    train_loader, _ = setup_data_loaders()
    if train_loader is None:
        print("Implement data loaders first.")
        return

    # Get first batch
    for features, labels in train_loader:
        break

    # Compute loss and gradients
    logits = model(features)
    loss = F.cross_entropy(logits, labels)

    if loss is not None:
        # Compute gradients
        loss.backward()

        max_grad_before = find_highest_gradient(model)
        print(f"Maximum gradient before clipping: {max_grad_before}\n")

        # TODO: Apply gradient clipping with max_norm=1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        max_grad_after = find_highest_gradient(model)
        print(f"Maximum gradient after clipping: {max_grad_after}")

        if max_grad_before is not None and max_grad_after is not None:
            print(f"\nGradient reduction: {max_grad_before / max_grad_after:.2f}x")


def train_with_all_techniques():
    """
    Combine all techniques: warmup, cosine decay, and gradient clipping in training.
    """

    print("Training with All Techniques\n")

    # Setup
    torch.manual_seed(123)
    train_loader, test_loader = setup_data_loaders()

    # Training hyperparameters
    n_epochs = 5
    initial_lr = 0.00001
    peak_lr = 0.01
    min_lr = 0.1 * initial_lr

    # Create model and optimizer
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)

    # Calculate training steps
    total_steps = len(train_loader) * n_epochs
    warmup_steps = int(0.2 * total_steps)

    # Calculate learning rate increment
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    global_step = -1
    train_losses = []
    track_lrs = []

    print(f"Training for {n_epochs} epochs")
    print(f"Warmup steps: {warmup_steps}/{total_steps}\n")

    for epoch in range(n_epochs):
        model.train()

        for batch_idx, (features, labels) in enumerate(train_loader):
            global_step += 1

            # Implement learning rate schedule
            if global_step < warmup_steps:
                # Phase 1: Warmup
                lr = initial_lr + global_step * lr_increment
            else:
                # Phase 2: Cosine annealing
                progress = ((global_step - warmup_steps) /
                            (total_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                        1 + math.cos(math.pi * progress))

            # Apply learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)

            # Forward pass
            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (after warmup)
            if global_step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            train_losses.append(loss.item())

            if global_step % 5 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch + 1}/{n_epochs} | Step {global_step:3d} | "
                      f"Loss: {loss:.4f} | LR: {current_lr:.6f}")

    train_acc = compute_accuracy(model, train_loader)
    test_acc = compute_accuracy(model, test_loader)
    print(f"\nFinal Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

    # Plot learning rate schedule
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(track_lrs)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Learning Rate")
    ax1.set_title("Learning Rate Schedule")
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_losses)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return model


if __name__ == "__main__":

    # A.1 Setup
    check_pytorch_setup()

    # A.2 Tensors
    create_tensors()
    numpy_tensor_conversion()
    tensor_data_types()
    tensor_operations()

    # A.3 Computation Graphs
    computation_graph()

    # A.4 Automatic Differentiation
    automatic_differentiation()

    # A.5 Neural Networks
    create_and_inspect_model()

    # A.6 Data Loaders
    setup_data_loaders()

    # A.7 Training
    train_model()

    # A.8 Evaluation
    evaluate_model()

    # A.9 Saving/Loading
    save_and_load_model()

    # D. Learning Rate Warmup
    learning_rate_warmup()

    # D. Cosine Annealing
    cosine_annealing()

    # D. Gradient Clipping
    gradient_clipping()

    # D. Train with all techniques
    train_with_all_techniques()