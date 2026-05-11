def split_params_for_muon(model):
    """
    Split model parameters into:
      - muon_params:  2D weight matrices of inner linear layers
      - adamw_params: everything else (embeddings, norms, biases, pos_embed, etc.)

    Returns (muon_params, adamw_params).
    """
    muon_params  = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (param.ndim == 2
                and 'embed'     not in name
                and 'norm'      not in name
                and 'pos_embed' not in name):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"Muon params:  {len(muon_params)}")
    print(f"AdamW params: {len(adamw_params)}")
    return muon_params, adamw_params
