
def validate_dimensions(dim: int, scale: float, operation: str = 'multiply') -> int:
    """Validate and compute scaled dimensions safely"""
    if scale <= 0:
        raise ValueError(f"Scale factor must be positive, got {scale}")
        
    result = int(dim * scale) if operation == 'multiply' else int(dim / scale)
    
    if result <= 0:
        raise ValueError(f"Invalid dimension after scaling: {result}")
        
    return result
