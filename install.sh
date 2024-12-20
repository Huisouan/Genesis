#!/bin/bash

# 检查是否提供了参数
if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2;
    print_help
    exit 1
fi

# Exits if error occurs
set -e

# 获取当前脚本所在目录
export GENESIS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Helper functions
extract_python_exe() {
    # 使用 conda 的 python 如果存在
    if ! [[ -z "${CONDA_PREFIX}" ]]; then
        local python_exe=${CONDA_PREFIX}/bin/python
    else
        local python_exe=$(which python3)
    fi
    if [ ! -f "${python_exe}" ]; then
        echo -e "[ERROR] Unable to find any Python executable at path: '${python_exe}'" >&2
        exit 1
    fi
    echo ${python_exe}
}

install_extension() {
    local python_exe=$(extract_python_exe)
    if [ -f "$1/setup.py" ]; then
        echo -e "\t module: $1"
        ${python_exe} -m pip install --editable $1
    fi
}

setup_conda_env() {
    local env_name=$1
    if ! command -v conda &> /dev/null; then
        echo "[ERROR] Conda could not be found. Please install conda and try again."
        exit 1
    fi
    if { conda env list | grep -w ${env_name}; } >/dev/null 2>&1; then
        echo -e "[INFO] Conda environment named '${env_name}' already exists."
    else
        echo -e "[INFO] Creating conda environment named '${env_name}'..."
        conda create -y --name ${env_name} python=3.10
    fi
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${env_name}
    conda deactivate
    echo -e "[INFO] Created conda environment named '${env_name}'.\n"
    echo -e "\t\t1. To activate the environment, run:                conda activate ${env_name}"
    echo -e "\t\t2. To install Genesis extensions, run:            genesis -i"
    echo -e "\t\t4. To perform formatting, run:                      genesis -f"
    echo -e "\t\t5. To deactivate the environment, run:              conda deactivate"
    echo -e "\n"
}

update_vscode_settings() {
    echo "[INFO] Setting up vscode settings..."
    local python_exe=$(extract_python_exe)
    local setup_vscode_script="${GENESIS_PATH}/.vscode/tools/setup_vscode.py"
    if [ -f "${setup_vscode_script}" ]; then
        ${python_exe} "${setup_vscode_script}"
    else
        echo "[WARNING] Unable to find the script 'setup_vscode.py'. Aborting vscode settings setup."
    fi
}

run_pre_commit() {
    if [ ! command -v pre-commit &>/dev/null ]; then
        echo "[INFO] Installing pre-commit..."
        pip install pre-commit
    fi
    echo "[INFO] Formatting the repository..."
    cd ${GENESIS_PATH}
    pre-commit run --all-files
    cd - > /dev/null
}

run_docker_container() {
    local docker_script=${GENESIS_PATH}/docker/container.sh
    echo "[INFO] Running docker utility script from: ${docker_script}"
    bash ${docker_script} $@
}

build_docs() {
    echo "[INFO] Building documentation..."
    local python_exe=$(extract_python_exe)
    cd ${GENESIS_PATH}/docs
    ${python_exe} -m pip install -r requirements.txt > /dev/null
    ${python_exe} -m sphinx -b html -d _build/doctrees . _build/html
    echo -e "[INFO] To open documentation on default browser, run:"
    echo -e "\n\t\txdg-open $(pwd)/_build/html/index.html\n"
    cd - > /dev/null
}

print_help() {
    echo -e "\nusage: $(basename "$0") [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-c] -- Utility to manage Genesis."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help           Display the help content."
    echo -e "\t-i, --install      Install the extensions inside Genesis."
    echo -e "\t-f, --format         Run pre-commit to format the code and check lints."
    echo -e "\t-p, --python         Run the python executable provided by Genesis or virtual environment (if active)."
    echo -e "\t-s, --sim            Run the simulator executable provided by Genesis."
    echo -e "\t-t, --test           Run all python unittest tests."
    echo -e "\t-o, --docker         Run the docker container helper script."
    echo -e "\t-v, --vscode         Generate the VSCode settings file from template."
    echo -e "\t-d, --docs           Build the documentation from source using sphinx."
    echo -e "\t-c, --conda [NAME]   Create the conda environment for Genesis. Default name is 'genesis'."
    echo -e "\n" >&2
}

# Main
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--install)
            echo "[INFO] Installing extensions inside the Genesis repository..."
            local python_exe=$(extract_python_exe)
            export -f extract_python_exe
            export -f install_extension
            find -L "${GENESIS_PATH}/extensions" -mindepth 1 -maxdepth 1 -type d -exec bash -c 'install_extension "{}"' \;
            unset extract_python_exe
            unset install_extension
            shift
            ;;
        -c|--conda)
            if [ -z "$2" ]; then
                echo "[INFO] Using default conda environment name: genesis"
                conda_env_name="genesis"
            else
                echo "[INFO] Using conda environment name: $2"
                conda_env_name=$2
                shift
            fi
            setup_conda_env ${conda_env_name}
            shift
            ;;
        -f|--format)
            run_pre_commit
            shift
            ;;
        -p|--python)
            local python_exe=$(extract_python_exe)
            echo "[INFO] Using python from: ${python_exe}"
            shift
            ${python_exe} $@
            break
            ;;
        -s|--sim)
            local sim_exe="${GENESIS_PATH}/simulator/simulator.sh"
            echo "[INFO] Running simulator from: ${sim_exe}"
            shift
            ${sim_exe} $@
            break
            ;;
        -t|--test)
            local python_exe=$(extract_python_exe)
            shift
            ${python_exe} ${GENESIS_PATH}/tests/run_all_tests.py $@
            break
            ;;
        -o|--docker)
            run_docker_container $@
            shift
            ;;
        -v|--vscode)
            update_vscode_settings
            shift
            ;;
        -d|--docs)
            build_docs
            shift
            ;;
        -h|--help)
            print_help
            exit 1
            ;;
        *)
            echo "[Error] Invalid argument provided: $1"
            print_help
            exit 1
            ;;
    esac
done