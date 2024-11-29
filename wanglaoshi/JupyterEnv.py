import sys
import os

def run_cmd(cmd):
    """
    Run command in the system.
    :param cmd: command to run
    :return: None
    """
    print('Running command:', cmd)
    p = os.popen(cmd)
    results = p.read()    #使用read()获取到一个字符串
    results_lst = results.split('\n')
    for line in results_lst:
        print(line)

def running_os():
    """
    Return the running environment.
    """
    env = sys.platform
    if env == 'win32':
        return 'Windows'
    elif env == 'linux':
        return 'Linux'
    elif env == 'darwin':
        return 'MacOS'
    else:
        return 'Unknown'

def running_python_version():
    """
    Return the running Python version.
    :return:
    """
    return sys.version

def running_python_path():
    """
    Return the running Python path.
    :return:
    """
    return sys.executable

def running_python_version_info():
    """
    Return the running Python version info.
    :return:
    """
    return sys.version_info

def running():
    """
    Return the running environment, Python version, Python path and Python version info.
    :return:
    """
    print("*"*40)
    print('Running environment:', running_os())
    print('Running Python version:', running_python_version())
    print('Running Python path:', running_python_path())
    print('Running Python version info:', running_python_version_info())
    print("*" * 40)
    return running_os(), running_python_version(), running_python_path(), running_python_version_info()

def install_ipykernel():
    """
    Install ipykernel.
    """
    # !pip install ipykernel
    run_cmd('pip install ipykernel')

def jupyter_kernel_list():
    """
    List the installed Jupyter kernels.
    """
    # !jupyter kernelspec list
    run_cmd('jupyter kernelspec list')

def install_kernel():
    """
    Install the kernel.
    """
    # !python -m ipykernel install --user --name=python3
    jupyter_kernel_list()
    install_ipykernel()
    running()
    python_path = running_python_path()
    print("Please input the new kernel name and display name.")
    kernel_name = input("Please input the kernel name:")
    kernel_display_name = input("Please input the kernel display name:")
    run_cmd(python_path + ' -m ipykernel install --user --name=' + kernel_name + ' --display-name=' + kernel_display_name)
