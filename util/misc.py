def process_machine_path(config):
    if config.machine == "g3":
        config.train.root = "/data1/jjc/experiment"
    elif config.machine == "ks":
        config.train.root = "/phd/jinjiachun/experiment"
    else:
        raise ValueError(f"Invalid machine: {config.machine}")

    return config