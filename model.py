import torch
import torch.nn as nn

def build_cnn_from_architecture(architecture, in_channels=3, num_classes=10):
    """Builds a simple CNN model from a given architecture definition."""
    layers = []
    current_in_channels = in_channels

    for op_def in architecture:
        op_type = op_def['op']
        params = op_def['params']

        if 'conv' in op_type:
            out_channels = params.get('out_channels', current_in_channels)
            kernel_size = int(op_type.split('_')[1][0]) # e.g., '3' from 'conv_3x3'
            padding = kernel_size // 2 # 'same' padding
            layers.append(nn.Conv2d(current_in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU())
            current_in_channels = out_channels
        elif 'pool' in op_type:
            kernel_size = int(op_type.split('_')[2][0]) # e.g., '2' from 'max_pool_2x2'
            if 'max' in op_type:
                layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size))
            elif 'avg' in op_type:
                layers.append(nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size))

    # Add a final pooling and linear layer for classification
    layers.append(nn.AdaptiveAvgPool2d((1, 1))) # pool to 1x1 feature map
    layers.append(nn.Flatten())
    layers.append(nn.Linear(current_in_channels, num_classes))

    return nn.Sequential(*layers)

if __name__ == '__main__':
    # Example usage
    sample_architecture = [
        {'op': 'conv_3x3', 'params': {'out_channels': 16}},
        {'op': 'max_pool_2x2', 'params': {}},
        {'op': 'conv_5x5', 'params': {'out_channels': 32}},
        {'op': 'avg_pool_2x2', 'params': {}},
        {'op': 'conv_3x3', 'params': {'out_channels': 64}}
    ]

    model = build_cnn_from_architecture(sample_architecture, in_channels=3, num_classes=10)
    print(model)

    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 32, 32) # Batch x Channels x Height x Width
    output = model(dummy_input)
    print("Output shape:", output.shape) # Should be (1, num_classes)
