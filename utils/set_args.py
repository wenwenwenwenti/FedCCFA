import argparse
import yaml

def set_arguments_with_yaml(yaml_path):
    '''使用对原有的yaml文件保留为默认参数, 运行程序时输入的其他参数为输入参数, 并返回和读取yaml同样的数据类型dict'''
    with open(yaml_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    parser = argparse.ArgumentParser(description="Run Federated Learning with customizable parameters.")
    for key, value in config.items():
        arg_type = type(value)
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", type=arg_type, default=value, help=f"(default: {value})", choices=[True, False])
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=value, help=f"(default: {value})")
    
    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict