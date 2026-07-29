"""Backtest the 5-minute Short MFE H3 strategy.

The strategy mirrors the Long implementation:
- Filter on minute 1 when price falls by ``filter_take_profit``.
- Enter Short from the original H1 open reference.
- From H1 minutes 2-5 through H3, TP uses favorable downside movement and
  SL uses adverse upside movement.
- Exit unresolved trades at close H3 using ``1 - close / entry``.
- Optimize TP/SL on Val and apply the selected pair unchanged to Test.

PowerShell:
    python -m temp.backtest_5m_short_mfe_fixed_tp_sl_h3 `
      --archive crypto/results/crypto_btc_short_mfe_h3_tp01_top40_seed1_8h.json `
      --rank 1 `
      --top-fraction 0.40 `
      --filter-take-profit 0.00025 `
      --take-profit 0.5 `
      --stop-loss 0.5 `
      --trade-cost 0.00016 `
      --same-candle-policy stop_first `
      --next-1m-tp-filter `
      --data data/crypto/BTCUSDT_5m.csv `
      --data-1m data/crypto/BTCUSDT_1m.csv `
      --out-dir temp/output
"""

from pathlib import Path

from temp.backtest_5m_long_mfe_fixed_tp_sl_h3 import main


DEFAULT_SHORT_ARCHIVE = Path(
    "crypto/results/crypto_btc_short_mfe_h3_tp01_top40_seed1_8h.json"
)


if __name__ == "__main__":
    main(
        default_archive=DEFAULT_SHORT_ARCHIVE,
        default_direction="short",
    )
