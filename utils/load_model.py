import pickle

def load_model(model, filepath):
    """
    Load model from output folder 

    Args:
        filepath: location of mdoel pkl file

    returns:
        model: trained model
    """
    
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    
    with open(filepath, 'rb') as file:
        trained_model = pickle.load(file)
    
    return trained_model