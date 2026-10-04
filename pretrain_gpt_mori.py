"""pretrain_gpt.py with every MoELayer's token dispatcher replaced by the MORI dispatcher
(megatron/core/transformer/moe/mori_dispatcher.py)."""
import time
from functools import partial

import pretrain_gpt as pg  # noqa: E402
from megatron.core.transformer.moe.mori_dispatcher import install_mori_dispatcher  # noqa: E402


def mori_builder(*args, **kwargs):
    model = pg.gpt_builder(*args, **kwargs)
    n = install_mori_dispatcher(model)
    print(f"[mori] replaced {n} MoE token dispatchers", flush=True)
    return model


if __name__ == "__main__":
    pg.set_startup_timestamps(program_start=pg._PROGRAM_START_TIME, main_entry=time.time())
    pg.train_valid_test_datasets_provider.is_distributed = True
    pretrain, store = pg.inprocess_restart.maybe_wrap_for_inprocess_restart(pg.pretrain)
    pretrain(
        pg.train_valid_test_datasets_provider,
        partial(pg.model_provider, mori_builder),
        pg.ModelType.encoder_or_decoder,
        pg.forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=pg.combined_args_provider,
        store=store,
        get_embedding_ranks=pg.get_embedding_ranks,
    )
