from transformer_engine.common.recipe import MXFP4BlockScaling


def get_fp4_recipe(config):
    """Returns the FP4 recipe. Can be monkey-patched for healing."""
    return MXFP4BlockScaling()
