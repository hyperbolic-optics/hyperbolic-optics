# LiNbO₃ nanofins with a tilted optic axis (arXiv:2608.06185, Sec. 3A)

A crossed-grating case beyond FMMAX: the optic axis of the LiNbO₃ fins is
tilted 45° from the layer normal in the xz plane, so ε has an xz component.

- εₒ = 5.46, εₑ = 5.06 (non-dispersive, as in the paper)
- fins 0.2 × 0.1 µm, 0.3 µm tall, on a 0.34 µm square lattice
- SiO₂ substrate (n = 1.46), normal incidence from air, 550 nm

```bash
uv run python benchmarks/crossed_grating/lnbo3_nanofin.py
```

It takes about a minute and writes `output/lnbo3_nanofin.png` (not committed).

## Result

T_xx for the long side along x, at 550 nm:

| harmonics | normal-vector rule | Laurent's rule |
|---|---|---|
| 49 | 0.9558 | 0.9419 |
| 121 | 0.9501 | 0.9450 |
| 289 | 0.9520 | 0.9464 |
| paper | 0.9496 (BCM), 0.9483 (FEM) | |

The two rules approach from opposite sides and bracket the paper's values, to
within 0.4%. The paper's own two methods differ by 0.13%.

Panel (b) sweeps the tilt from 0° to 90°. For this weakly birefringent crystal
(Δε = 0.4) the fin's shape dominates, and T_xx moves by only 0.002. The mirror
y → −y maps both the fin and the axis to themselves, so x → y conversion must
vanish at every tilt; the script prints it (1e-27).
