"""
Patch for fairseq to work with PyTorch 2.6+ where weights_only=True by default
"""
import torch
import fairseq.checkpoint_utils

# Save the original function
_original_load_checkpoint_to_cpu = fairseq.checkpoint_utils.load_checkpoint_to_cpu

def patched_load_checkpoint_to_cpu(path, arg_overrides=None, load_on_all_ranks=False):
    """
    Patched version that sets weights_only=False for compatibility
    """
    # Get the original function's code
    import inspect
    import types
    
    # Get source code and modify the torch.load line
    source = inspect.getsource(_original_load_checkpoint_to_cpu)
    
    # Create a modified version that uses weights_only=False
    modified_source = source.replace(
        'torch.load(f, map_location=torch.device("cpu"))',
        'torch.load(f, map_location=torch.device("cpu"), weights_only=False)'
    )
    
    # Compile and execute the modified function
    code = compile(modified_source, '<string>', 'exec')
    namespace = {
        'torch': torch,
        'PathManager': fairseq.checkpoint_utils.PathManager,
        'logger': fairseq.checkpoint_utils.logger,
        'os': fairseq.checkpoint_utils.os,
        'IOError': IOError,
    }
    exec(code, namespace)
    
    # Call the modified function
    return namespace['load_checkpoint_to_cpu'](path, arg_overrides, load_on_all_ranks)

# Apply the patch
fairseq.checkpoint_utils.load_checkpoint_to_cpu = patched_load_checkpoint_to_cpu