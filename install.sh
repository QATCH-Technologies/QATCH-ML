#!/bin/bash

# Function to check Python version
check_python_version() {
    python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$python_version" < "3.9" || "$python_version" > "3.12" ]]; then
        echo "Error: Python version 3.9–3.12 is required. Current version: $python_version"
        exit 1
    else
        echo "Python version is $python_version - OK"
    fi
}

# Function to check pip version
check_pip_version() {
    pip_version=$(pip --version | awk '{print $2}')
    os=$(uname -s)

    case $os in
        Linux)
            required_pip_version="19.0"
            ;;
        Darwin)
            required_pip_version="20.3"
            ;;
        *)
            required_pip_version="19.0"
            ;;
    esac

    if [[ "$(echo "$pip_version < $required_pip_version" | bc)" -eq 1 ]]; then
        echo "Error: pip version $required_pip_version or higher is required for $os. Current version: $pip_version"
        exit 1
    else
        echo "pip version is $pip_version - OK"
    fi
}

# Function to display NVIDIA and Windows-specific requirements
display_additional_requirements() {
    echo ""
    echo "For GPU support, ensure the following NVIDIA software is installed:"
    echo "  - NVIDIA GPU drivers >= 525.60.13 for Linux"
    echo "  - NVIDIA GPU drivers >= 528.33 for WSL on Windows"
    echo "  - CUDA Toolkit 12.3"
    echo "  - cuDNN SDK 8.9.7"
    echo "  - (Optional) TensorRT for inference optimization"
    echo ""
    echo "For Windows Native, ensure Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, and 2019 is installed."
    echo "Download it from: https://aka.ms/vs/16/release/vc_redist.x64.exe"
}

# Function to install Python dependencies
install_dependencies() {
    echo "Installing Python dependencies..."
    pip install matplotlib \
                "numpy<2.0" \
                scikit-learn \
                scipy \
                xgboost \
                tensorflow \
                tsmoothie \
                pandas \
                hyperopt \
                seaborn \
                tqdm

    echo "Installation of Python packages complete!"
}

# Run the version checks
check_python_version
check_pip_version

# Create and activate virtual environment (optional)
python3 -m venv env
source env/bin/activate

# Install dependencies
install_dependencies

# Save installed packages to requirements.txt
pip freeze > requirements.txt

# Display additional requirements
display_additional_requirements

echo "Setup complete!"
