def process_machine_path(config):
    if config.machine == "g3":
        config.train.root = "/data1/jjc/experiment"
        config.janus_path = "/data1/ckpts/deepseek-ai_/Janus-Pro-1B"
    elif config.machine == "ks":
        config.train.root = "/phd/jinjiachun/experiment"
        config.janus_path = "/phd/jinjiachun/ckpt/deepseek-ai/Janus-Pro-1B"
    else:
        raise ValueError(f"Invalid machine: {config.machine}")

    return config