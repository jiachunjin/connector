def process_path_for_different_machine(config):
    if config.machine == "g3":
        config.train.root = "/data1/jjc/experiment"
        config.janus_path = "/data1/ckpts/deepseek-ai_/Janus-Pro-1B"
        config.data.train_path = "/data1/LargeData/timm/imagenet-1k-wds"
    elif config.machine == "ks":
        config.train.root = "/phd/jinjiachun/experiment"
        config.janus_path = "/phd/jinjiachun/ckpt/deepseek-ai/Janus-Pro-1B"
        config.data.train_path = "/phd/jinjiachun/dataset/timm/imagenet-1k-wds"
    else:
        raise ValueError(f"Invalid machine: {config.machine}")

    return config

def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)