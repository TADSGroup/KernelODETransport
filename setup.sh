#!/bin/bash

# Define global variable for environment name
ENV_NAME="kode_env_test"

# Function to check if Conda is installed
check_conda(){
  if ! command -V conda &> /dev/null; then
    echo "Conda could not be found. Please install conda before continuing."
    exit 1

  fi
}


# Creating a new Conda environment and activating it
create_conda_env(){
  echo "Creating a new Conda environment..."
  conda create --name "$ENV_NAME" python=3.8 -y
}


# Global variable for conda
ENV_BIN_PATH="$(conda info --base)/envs/$ENV_NAME/bin"

# Installing Python packages from requirements.txt
install_requirements(){
  echo "Installing requirements from requirements.txt in the conda
  environment $ENV_NAME..."
  "$ENV_BIN_PATH/pip" install -r requirements.txt
}

# Installing JAX based on CPU/GPU and cuda versions
install_jax(){

  # Identify the operating system and machine hardware name
  local os_name="$(uname -s)"
  local hw_name="$(uname -m)"

  echo "Detected OS: $os_name, Hardware: $hw_name"

  # Default to CPU Linux installation
  local jax_install_command="pip install --upgrade 'jax[cpu]'"

  # Check if Apple Silicon or Linux
  if [[ "$os_name" == "Darwin" && "$hw_name" == "arm64" ]]; then
    echo "Detected Apple MacBook Silicon M1"
    jax_install_command="pip install jax-metal"

  elif [[ "$os_name" == "Linux" ]]; then
    echo "Detected a Linux system"
    # Linux with or without GPU

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
      # Attempt to determine CUDA version
      local cuda_version=$(nvidia-smi | grep -oP 'CUDA Version: \K\d+')

      echo "Detected CUDA Version: ${cuda_version}"

      case "$cuda_version" in
        11)
            jax_install_command="pip install --upgrade 'jax[cuda11_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
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

  echo "Running JAX install command: $ENV_BIN_PATH/$jax_install_command"
  eval "$ENV_BIN_PATH/$jax_install_command"
  }


# Install Diffrax and optax
install_diffrax(){
  echo "Installing Diffrax for solving ODEs in JAX..."
  eval "$ENV_BIN_PATH/pip install diffrax optax"
}


# Main script execution
main(){
  check_conda
  create_conda_env
  install_requirements
  install_jax
  install_diffrax
  echo "Setup complete."
}

main
