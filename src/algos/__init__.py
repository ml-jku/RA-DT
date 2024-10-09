MODEL_CLASSES = {
    "DT": None,
    "ODT": None,
    "UDT": None,
    "DDT": None,
    "HelmDT": None,
    "DHelmDT": None,
    "CDT": None,
    "DCDT": None,
}

AGENT_CLASSES = {
    "DT": None,
    "ODT": None,
    "UDT": None,
    "DDT": None,
    "MDDT": None,
    "HelmDT": None,
    "DHelmDT": None,
    "CDT": None,  
    "DCDT": None,
}


def get_model_class(kind):
    if kind in ["DT", "ODT", "UDT"]:
        from .models.online_decision_transformer_model import OnlineDecisionTransformerModel
        MODEL_CLASSES[kind] = OnlineDecisionTransformerModel
    elif kind in ["DDT"]:
        from .models.discrete_decision_transformer_model import DiscreteDTModel
        MODEL_CLASSES[kind] = DiscreteDTModel
    elif kind in ["HelmDT"]:
        from .models.helm_decision_transformer_model import HelmDTModel
        MODEL_CLASSES[kind] = HelmDTModel
    elif kind in ["DHelmDT"]:
        from .models.helm_decision_transformer_model import DiscreteHelmDTModel
        MODEL_CLASSES[kind] = DiscreteHelmDTModel
    elif kind in ["CDT"]:
        from .models.cache_decision_transformer_model import CacheDTModel
        MODEL_CLASSES[kind] = CacheDTModel
    elif kind in ["DCDT"]:
        from .models.cache_decision_transformer_model import DiscreteCacheDTModel
        MODEL_CLASSES[kind] = DiscreteCacheDTModel
    assert kind in MODEL_CLASSES, f"Unknown kind: {kind}"
    return MODEL_CLASSES[kind]


def get_agent_class(kind):
    assert kind in AGENT_CLASSES, f"Unknown kind: {kind}"
    # lazy imports only when needed
    if kind in ["DT", "ODT", "HelmDT", "DHelmDT"]:
        from .decision_transformer_sb3 import DecisionTransformerSb3
        AGENT_CLASSES[kind] = DecisionTransformerSb3
    elif kind in ["UDT"]:
        from .universal_decision_transformer_sb3 import UDT
        AGENT_CLASSES[kind] = UDT
    elif kind in ["DDT"]:
        from .discrete_decision_transformer_sb3 import DiscreteDecisionTransformerSb3
        AGENT_CLASSES[kind] = DiscreteDecisionTransformerSb3
    elif kind == "CDT":
        from .cache_decision_transformer_sb3 import CacheDecisionTransformerSb3, DiscreteCacheDecisionTransformerSb3
        AGENT_CLASSES[kind] = CacheDecisionTransformerSb3
    elif kind == "DCDT": 
        from .cache_decision_transformer_sb3 import DiscreteCacheDecisionTransformerSb3
        AGENT_CLASSES[kind] = DiscreteCacheDecisionTransformerSb3
    return AGENT_CLASSES[kind]
