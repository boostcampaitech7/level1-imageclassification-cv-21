# model/model_factory.py
import importlib

def create_model(model_name, *args, **kwargs):
    model_module = importlib.import_module(f"model.{model_name}")
    model_class = getattr(model_module, model_name)
    model = model_class(*args, **kwargs)
    return model
