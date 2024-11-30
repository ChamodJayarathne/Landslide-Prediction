# def preprocess_data(rainfall, river_level):
#     # Example: Scale or normalize if necessary
#     return [rainfall, river_level]

# utils/preprocessing.py

def preprocess_data(rainfall, river_level):
    """
    Preprocess the input data for model prediction
    """
    try:
        # Convert inputs to float
        rainfall = float(rainfall)
        river_level = float(river_level)
        
        # Basic validation
        if rainfall < 0 or river_level < 0:
            raise ValueError("Rainfall and river level must be non-negative")
            
        return [rainfall, river_level]
    except ValueError as e:
        raise ValueError(f"Invalid input values: {str(e)}")