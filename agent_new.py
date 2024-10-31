import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class TransformerWithMLP(nn.Module):
    """
    Initializes the TransformerWithMLP class as a PyTorch module.

    Parameters:
    - input_dim: int, the dimension of the input features.
    - output_dim: int, the dimension of the output features.
    - transformer_dim: int, the feature dimension used by the Transformer.
    - num_heads: int, the number of attention heads in the Transformer.
    - mlp_dim: int, the feature dimension used by the MLP layers.
    - num_layers: int, the number of layers in the Transformer encoder.
    - mlp_layers: int, the number of layers in the MLP.
    - dropout_rate: float, the dropout rate used in both the Transformer and MLP for regularization.

    This constructor initializes the Transformer and MLP components with the specified parameters.
    """
    def __init__(self, input_dim, output_dim, transformer_dim, num_heads, mlp_dim, num_layers, mlp_layers, dropout_rate=0.1):
        super(TransformerWithMLP, self).__init__()
        # Fully connected layer to transform input dimension to transformer dimension
        self.fc1 = nn.Linear(input_dim, transformer_dim)

        # Define a single Transformer encoder layer with specified features and dropout
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)

        # Stack multiple Transformer encoder layers to form the encoder
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers)

        # Build the MLP layers
        mlp_modules = [nn.Linear(transformer_dim, mlp_dim), nn.ReLU(), nn.LayerNorm(mlp_dim)]
        for _ in range(mlp_layers - 2):
            mlp_modules.extend([
                nn.Linear(mlp_dim, mlp_dim), # Additional MLP layers
                nn.ReLU(), # ReLU activation
                nn.LayerNorm(mlp_dim),  # Layer normalization
                nn.Dropout(dropout_rate)  # Dropout for regularization
            ])
        mlp_modules.append(nn.Linear(mlp_dim, output_dim)) # Final layer to output dimension
        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, x):
        """
        Defines the forward pass of the TransformerWithMLP model.

        Parameters:
        - x: torch.Tensor, the input tensor.

        Returns:
        - x: torch.Tensor, the output tensor after passing through Transformer and MLP layers.
        """
        # Flatten input if it's not already flat
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # 如果输入是多维的，需要展平

        # Pass input through the first fully connected layer
        x = self.fc1(x)

        # Add an extra dimension for Transformer compatibility
        x = x.unsqueeze(1)

        # Pass through the Transformer encoder
        x = self.transformer_encoder(x)

        # Remove the extra dimension after Transformer
        x = x.squeeze(1)

        # Pass through the MLP layers
        x = self.mlp(x)
        return x

class AI_Agent:

    def __init__(self, user_input):
        self.user_input = user_input

    def prepare_data(self, file_path, input_columns_range, output_columns_range, test_size, val_size, random_state):
        """
        Prepare the dataset for training.

        Parameters:
        - file_path: str, the path to the CSV file.
        - input_columns_range: tuple, the range of columns to use as input features.
        - output_columns_range: tuple, the range of columns to use as output targets.
        - test_size: float, the proportion of the dataset to include in the test split.
        - val_size: float, the proportion of the training set to include in the validation set.
        - random_state: int, the seed used by the random number generator.

        Returns:
        - X_train: Training features.
        - X_val: Validation features.
        - X_test: Test features.
        - y_train: Training target.
        - y_val: Validation target.
        - y_test: Test target.
        """
        # load data
        data = pd.read_csv(file_path, encoding='ISO-8859-1')

        # Extract input and output data
        X = data.iloc[:, input_columns_range[0]:input_columns_range[1]].values
        y = data.iloc[:, output_columns_range[0]:output_columns_range[1]].values

        # Normalize
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_normalized = scaler_X.fit_transform(X)
        y_normalized = scaler_y.fit_transform(y)

        # split dataset
        X_train, X_temp, y_train, y_temp = train_test_split(X_normalized, y_normalized, test_size=test_size,
                                                            random_state=random_state)
        val_test_size = val_size / (1 - test_size)  # Adjust validation size based on remaining data after test split
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_test_size, random_state=random_state)

        # Remove rows with NaN values
        # For each dataset (train, validation, test), remove rows where any element is NaN

        def remove_nan_rows(X, y):
            """Helper function to remove rows with NaN values."""
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
            return X[mask], y[mask]

        X_train, y_train = remove_nan_rows(X_train, y_train)
        X_val, y_val = remove_nan_rows(X_val, y_val)
        X_test, y_test = remove_nan_rows(X_test, y_test)

        return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y

    def predict_and_evaluate(self, input_data, model, weights, scaler_X, scaler_y, device):
        """
        Load model weights, predict output values for user-provided input, and compare with ground truth.

        Parameters:
        - model: the trained model.
        - scaler_X: scaler used to normalize input data.
        - scaler_y: scaler used to inverse transform model output.
        - dataset: DataFrame containing the dataset for finding ground truth.
        """
        # Load the model weights
        model.load_state_dict(weights)
        model.eval()

        # Normalize input data
        input_normalized = scaler_X.transform(input_data)
        input_tensor = torch.FloatTensor(input_normalized).to(device)
        input_tensor.requires_grad = True  # Enable gradient calculation

        # Predict without changing the model to train mode
        with torch.no_grad():
            output1 = model(input_tensor)

        # Enable gradient calculation for prediction manually
        output = model(input_tensor)

        # Initialize a list to store gradients for each output
        all_gradients = []

        # Compute gradients for each output w.r.t input coordinates
        for i in range(output.shape[1]):  # Iterate over each output value
            grad_outputs = torch.zeros_like(output)
            grad_outputs[:, i] = 1  # Isolate gradient calculation for the i-th output

            gradients = torch.autograd.grad(outputs=output, inputs=input_tensor,
                                            grad_outputs=grad_outputs, create_graph=False, retain_graph=True)[0]

            # Extract gradients for x, y, z and append to the list
            all_gradients.append(gradients[:, :3])

        # Convert list of tensors to a single numpy array
        gradients_np = torch.cat(all_gradients, dim=1).cpu().numpy()

        output_np = output1.detach().cpu().numpy()
        predicted_values = scaler_y.inverse_transform(output_np)

        # Output
        predicted_values = predicted_values.flatten()
        gradients_np = gradients_np.flatten()

        output = np.concatenate((predicted_values, gradients_np), axis=0)

        print(f"Predicted Values: {predicted_values.flatten()}")
        print(f"Gradients: {gradients_np.flatten()}")

        return output

    def process_and_predict(self):
        """
        Process user input for the filename and input values, then predict the corresponding data.

        Parameters:
        - user_input (str): A string entered by the user in the format "filename,value1,value2,...,value11"

        Returns:
        - No return value, but prints the prediction results
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameter settings
        input_dim = 11  # Input dimension
        output_dim = 3  # Output dimension
        transformer_dim = 64  # Dimension of Transformer layers
        num_heads = 4  # Number of attention heads
        mlp_dim = 64  # Dimension of MLP layers
        num_layers = 2  # Number of Transformer encoder layers
        mlp_layers = 2  # Number of MLP layers
        learning_rate = 0.001  # Learning rate

        # Split the input string by commas
        try:
            base_filename, input_str = self.user_input.split(",", 1)  # Split only at the first comma
            base_filename = base_filename.strip()  # Remove leading/trailing whitespace from the filename
            input_str = input_str.strip()  # Remove leading/trailing whitespace from the input values
        except ValueError:
            print("Incorrect number of values entered.")
            return

        # Load data
        file_path = f'data/{base_filename}.csv'

        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = self.prepare_data(
            file_path=file_path,
            input_columns_range=(0, 11),
            output_columns_range=(15, 19),
            test_size=0.4,
            val_size=0.5,
            random_state=42
        )

        # Process input data
        try:
            input_list = input_str.split(',')
            if len(input_list) != 11:  # Check if there are exactly 11 elements
                raise ValueError("Incorrect number of values entered.")
            input_data = np.array([float(num) for num in input_list]).reshape(1, -1)
        except ValueError as e:
            print("Please enter valid values or format. Error:", e)
            return

        # Initialize model
        model = TransformerWithMLP(input_dim, output_dim, transformer_dim, num_heads, mlp_dim, num_layers, mlp_layers)
        model.to(device)

        # Load model weights and perform prediction
        weights = torch.load(f'model_test/{base_filename}.pth', map_location = torch.device('cpu'))
        output = self.predict_and_evaluate(input_data, model, weights, scaler_X, scaler_y, device)
        return output

# if __name__ == '__main__':
#
#     # Provide the user_input parameter to create an instance of the class
#     user_input = "No.1-1A,81.034207,-92.473084,-133.463402,1,0,0,0,0,0,0,0"
#     agent = AI_Agent(user_input)
#
#     # Call the method through the class instance
#     agent.process_and_predict()