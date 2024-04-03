#!/bin/bash

# Define global variables
ENV_NAME="kode_env"
OS_NAME="$(uname -s)"
HARDWARE_NAME="$(uname -m)"

echo "Detected OS: $OS_NAME, Hardware: $HARDWARE_NAME."
echo

# Function to check if Conda is installed
check_conda(){
  if ! command -V conda &> /dev/null; then
    echo "Conda could not be found. Please install Anaconda before
    continuing. https://www.anaconda.com/download"
    exit 1

  fi
}


# Creating a new Conda environment and activating it
create_conda_env(){
  echo "Creating a new Conda environment..."
  conda create --name "$ENV_NAME" python=3.9 -y
  echo

}


# Global variable for conda
ENV_BIN_PATH="$(conda info --base)/envs/$ENV_NAME/bin"


# Installing Python packages from requirements.txt
install_requirements(){
  echo "Installing requirements from requirements.txt in the conda
  environment $ENV_NAME..."
  "$ENV_BIN_PATH/pip" install -r requirements.txt
  echo
}


# Installing Pytorch based on Apple M1/Linux
install_pytorch(){

  # Default installation command
  local torch_install_command="pip install torch==1.10.1 torchvision"

    if [[ "$OS_NAME" == "Darwin" && "$HARDWARE_NAME" == "arm64" ]]; then

      # Update the mpmath to 1.3.0 for torch. Issue for M1 silicon.
      $ENV_BIN_PATH/pip install --upgrade mpmath==1.3.0
      torch_install_command="pip install --pre torch torchvision torchaudio
      --extra-index-url https://download.pytorch.org/whl/nightly/cpu"

    fi
  echo "Running TORCH install command: $ENV_BIN_PATH/$torch_install_command"
  eval "$ENV_BIN_PATH/$torch_install_command"
  echo
}


# Installing JAX based on Apple M1/Linux and CPU/GPU
install_jax(){

  # Default to CPU Linux installation
  local jax_install_command="pip install --upgrade 'jax[cpu]'"

  # Check if Apple Silicon or Linux
  if [[ "$OS_NAME" == "Darwin" && "$HARDWARE_NAME" == "arm64" ]]; then
    echo "Detected Apple MacBook Silicon M1"
    jax_install_command="pip install jax-metal"

  elif [[ "$OS_NAME" == "Linux" ]]; then
    echo "Detected a Linux system"
    # Linux with or without GPU

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
      # Attempt to determine CUDA version
      local cuda_version=$(nvidia-smi | grep -oP 'CUDA Version: \K\d+')

      echo "Detected CUDA Version: ${cuda_version}"

      case "$cuda_version" in
        11)
            jax_install_command="pip install jax[cuda11_pip]==0.3.10 \
            jaxlib==0.3.10+cuda11.cudnn82 -f \
            https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
            ;;

        12)
            jax_install_command="pip install --upgrade 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
            ;;
        *)
            echo "Unsupported CUDA version for JAX installation or CUDA not
            found. Falling back to CPU version."
            ;;
      esac
    else
        echo "No NVIDIA GPU detected. Installing JAX for CPU."
    fi
  fi

  echo
  echo "Running JAX install command: $ENV_BIN_PATH/$jax_install_command"
  eval "$ENV_BIN_PATH/$jax_install_command"
  echo
  }


# Install Diffrax
install_diffrax(){
  echo "Installing Diffrax for solving ODEs in JAX..."
  eval "$ENV_BIN_PATH/pip install diffrax==0.2.1 optax==0.1.1 equinox==0.8.0 \
   chex==0.1.3 jaxtyping==0.2.7"
  echo
}


# Main script execution
main(){
  check_conda
  create_conda_env
  install_requirements
  install_pytorch
  install_jax
  install_diffrax
  echo "Setup complete."
}

main