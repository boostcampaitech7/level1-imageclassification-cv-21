# Getting Started
===============

This project requires specific dependencies installed in your Python environment. Please follow the steps below to set up the environment and execute the train.py file.

## Step 1: Install Required Dependencies
First, install the required dependencies using pip:

```pip install -U lightning "ray[data,train,tune,serve]" wandb```
This will install the necessary dependencies, including PyTorch Lightning, Ray, and Weights & Biases.

## Step 2: Set up Weights & Biases
Make sure to set up your Weights & Biases account and install the SDK. You can find more information on setting up Weights & Biases [here](https://docs.wandb.ai/ko/quickstart).

## Step 3: Run the train.py File
Once the dependencies are installed, you can execute the train.py file using Python:

### Default Configuration
If you don't specify any arguments, the script will use the default configuration:

```python train.py```
This will train the model with the default settings.

### Custom Configuration
Alternatively, you can specify custom arguments to override the default settings:

```python train.py --model_name <model_name> --num_gpus <num_gpus> --smoke_test```
- --model_name: Specify the name of the model to train. You can choose from a list of available models by checking the model definitions in the codebase.Default model is ResNet18
- --num_gpus: Specify the number of GPUs to use for training. Use this when you train in multi-gpu environment. Default: 1
- --smoke_test: (Optional) If you want to run a quick smoke test to verify that the training script is working correctly, add this flag. The smoke test will run the training script with a small batch size and a limited number of epochs.

Example:
```python train.py --model_name resnet50 --num_gpus 2```
This will train the ResNet-50 model using 2 GPU.

Make sure to run this command from the directory where the train.py file is located.

## Troubleshooting
If you encounter any issues during the installation or execution of the train.py file, please check the following:

- Make sure you have the latest version of conda installed.
- Verify that the requirements.yaml file is in the correct location.
- Check that the conda environment is activated correctly.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.

## Contact
If you have any questions or need help with the project, please contact us at [insert contact information].