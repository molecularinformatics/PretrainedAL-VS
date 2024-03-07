"""This module contains the Model ABC and various implementations thereof. A
model is used to predict an input's objective function based on prior
training data."""

from typing import Optional, Set

import pytorch_lightning
from molpal.models.base import Model


def model(model: str, **kwargs) -> Model:
    """Model factory function"""
    if model == "rf":
        from molpal.models.sklmodels import RFModel

        return RFModel(**kwargs)

    if model == "lgbm":
        from molpal.models.sklmodels import LightGBMModel
        return LightGBMModel(**kwargs)

    if model == "gp":
        from molpal.models.sklmodels import GPModel

        return GPModel(**kwargs)

    if model == "nn":
        return nn(**kwargs)

    if model == "mpn":
        return mpn(**kwargs)
    
    if model == "transformer":
        return molformer(**kwargs)
    
    if model == "molclr":
        return clr(**kwargs)

    if model == "random":
        from molpal.models.random import RandomModel

        return RandomModel(**kwargs)

    raise NotImplementedError(f'Unrecognized model: "{model}"')


def nn(conf_method: Optional[str] = None, **kwargs) -> Model:
    """NN-type Model factory function"""
    from molpal.models.nnmodels import NNModel, NNDropoutModel, NNEnsembleModel, NNTwoOutputModel

    try:
        return {
            "dropout": NNDropoutModel,
            "ensemble": NNEnsembleModel,
            "twooutput": NNTwoOutputModel,
            "mve": NNTwoOutputModel,
            "none": NNModel,
        }.get(conf_method, "none")(conf_method=conf_method, **kwargs)
    except KeyError:
        raise NotImplementedError(f'Unrecognized NN confidence method: "{conf_method}"')


def mpn(conf_method: Optional[str] = None, **kwargs) -> Model:
    """MPN-type Model factory function"""
    from molpal.models.mpnmodels import MPNModel, MPNDropoutModel, MPNTwoOutputModel

    try:
        return {
            "dropout": MPNDropoutModel,
            "twooutput": MPNTwoOutputModel,
            "mve": MPNTwoOutputModel,
            "none": MPNModel,
        }.get(conf_method, "none")(conf_method=conf_method, **kwargs)
    except KeyError:
        raise NotImplementedError(f'Unrecognized MPN confidence method: "{conf_method}"')


def molformer(conf_method: Optional[str] = None, **kwargs) -> Model:
    """Pretrained Transformer Model factory function"""
    from molpal.models.transformermodels import TransformerModel, TransformerTwoOutputModel

    try:
        return {
            "twooutput": TransformerTwoOutputModel,
            "mve": TransformerTwoOutputModel,
            "none": TransformerModel,
        }.get(conf_method, "none")(conf_method=conf_method, **kwargs)
    except KeyError:
        raise NotImplementedError(f'Unrecognized Transformer confidence method: "{conf_method}"')


def clr(conf_method: Optional[str] = None, **kwargs) -> Model:
    """Pretrained MolCLR Model factory function"""
    from molpal.models.molclrmodels import MolCLRModel, MolCLRTwoOutputModel

    try:
        return {
            "twooutput": MolCLRTwoOutputModel,
            "mve": MolCLRTwoOutputModel,
            "none": MolCLRModel,
        }.get(conf_method, "none")(conf_method=conf_method, **kwargs)
    except KeyError:
        raise NotImplementedError(f'Unrecognized MolCLR confidence method: "{conf_method}"')



def model_types() -> Set[str]:
    return {"rf", "gp", "nn", "mpn", "lgbm", "transformer", "molclr"}
