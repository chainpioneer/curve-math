//! Full exchange simulation for TwoCryptoNG and TwoCryptoStable pools.
//!
//! Replicates on-chain `_exchange` + `tweak_price` exactly.
//! Uses `wad_exp` (Snekmate) for oracle EMA. Single price_scale (2-coin).
//! TwoCryptoStable uses the same tweak_price but StableSwap-based get_y.

use alloy_primitives::U256;

use crate::core::twocrypto_ng::{self, WAD, FEE_DENOMINATOR, A_MULTIPLIER, isqrt};
use crate::exchange_ng::wad_exp;

const N_COINS: u64 = 2;
const PRECISION: U256 = WAD;

/// Full mutable state for a TwoCryptoNG or TwoCryptoStable pool.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TwoCryptoNGState {
    pub balances: [U256; 2],
    pub price_scale: U256,
    pub price_oracle: U256,
    pub last_prices: U256,
    pub last_prices_timestamp: u64,
    pub d: U256,
    pub virtual_price: U256,
    pub xcp_profit: U256,
    /// `xcp_profit_a` from the on-chain pool — the cached value at the
    /// previous admin-fee claim. Used to compute `admin_share` exactly via
    /// `_claim_admin_fees` (`fees = (xcp_profit - xcp_profit_a) * admin_fee
    /// / 2e10`). Set by `_claim_admin_fees` to the new `xcp_profit` after a
    /// successful claim.
    pub xcp_profit_a: U256,
    /// Accumulator for admin LP fees collected during `add_liquidity`.
    /// On-chain `twocrypto_ng.vy`:
    ///     self.admin_lp_virtual_balance += ADMIN_FEE * d_token_fee / 10**10
    /// Consumed by `_claim_admin_fees` which uses it as the base for
    /// `admin_share` before adding the vprice-based contribution. NOT
    /// applicable to TwoCryptoStable (it starts admin_share from 0).
    /// Stored in the pool at storage slot 17 (non-public getter).
    pub admin_lp_virtual_balance: U256,
    pub ann: U256,
    pub gamma: U256,
    pub mid_fee: U256,
    pub out_fee: U256,
    pub fee_gamma: U256,
    pub precisions: [U256; 2],
    pub total_supply: U256,
    pub allowed_extra_profit: U256,
    pub adjustment_step: U256,
    /// Raw ma_time from packed_rebalancing_params (NOT the view function value).
    pub ma_time: U256,
    pub future_a_gamma_time: U256,
    /// If true, uses StableSwap invariant (TwoCryptoStable: MATH v0.1.0).
    pub is_stable: bool,
}

pub struct ExchangeResult {
    pub dy: U256,
    pub fee: U256,
}

// ── 2-coin get_p (analytical price derivative) ──────────────────────────────

/// Port of TwocryptoMath.get_p — returns dx0/dx1 in 1e18 precision.
fn get_p_2(xp: [U256; 2], d: U256, a_gamma: [U256; 2]) -> U256 {
    let p36 = U256::from(10u64).pow(U256::from(36u64));
    let p18 = WAD;

    // K0 = 4 * xp[0] * xp[1] / D * 10^36 / D
    let k0 = U256::from(4u64) * xp[0] * xp[1] / d * p36 / d;

    let gamma_plus_wad = a_gamma[1] + p18;
    let gk0 = U256::from(2u64) * k0 * k0 / p36 * k0 / p36
        + gamma_plus_wad * gamma_plus_wad
        - k0 * k0 / p36 * (U256::from(2u64) * a_gamma[1] + U256::from(3u64) * p18) / p18;

    // NNAG2 = N^N * A * gamma^2 / A_MULTIPLIER = 4 * A * gamma^2 / 10000
    let nnag2 = a_gamma[0] * a_gamma[1] * a_gamma[1] / A_MULTIPLIER;

    let denominator = gk0 + nnag2 * xp[0] / d * k0 / p36;

    xp[0] * (gk0 + nnag2 * xp[1] / d * k0 / p36) / xp[1] * p18 / denominator
}

/// Port of MATH v0.1.0 get_p — StableSwap price derivative.
/// Used by TwoCryptoStable pools.
fn get_p_2_stable(xp: [U256; 2], d: U256, a_gamma: [U256; 2]) -> U256 {
    let ann = a_gamma[0] * U256::from(N_COINS);
    let n_pow_n = U256::from(N_COINS).pow(U256::from(N_COINS)); // 4
    let mut dr = d / n_pow_n;
    for i in 0..2 {
        dr = dr * d / xp[i];
    }
    let xp0_a = ann * xp[0] / A_MULTIPLIER;
    WAD * (xp0_a + dr * xp[0] / xp[1]) / (xp0_a + dr)
}

/// Fee calculation for 2-coin CryptoSwap pool.
/// Fee for 2-coin pool.
/// `use_ng_formula`: true for TwoCryptoStable (v0.1.0), false for TwoCryptoNG (v2.x).
fn fee_2(xp: &[U256; 2], mid_fee: U256, out_fee: U256, fee_gamma: U256, use_ng_formula: bool) -> U256 {
    let f_sum = xp[0] + xp[1];
    let k = WAD * U256::from(4u64) * xp[0] / f_sum * xp[1] / f_sum;
    let f = if use_ng_formula {
        // NG formula: deployed in TwoCryptoStable (MATH v0.1.0) bytecode
        fee_gamma * k / (fee_gamma * k / WAD + WAD - k)
    } else {
        // V1 formula: deployed in TwoCryptoNG (MATH v2.x) bytecode
        fee_gamma * WAD / (fee_gamma + WAD - k)
    };
    (mid_fee * f + out_fee * (WAD - f)) / WAD
}

impl TwoCryptoNGState {
    pub fn exchange(&mut self, i: usize, j: usize, dx: U256, block_timestamp: u64) -> Option<ExchangeResult> {
        if i == j || i >= 2 || j >= 2 || dx.is_zero() {
            return None;
        }

        let a_gamma = [self.ann, self.gamma];
        let mut xp = self.balances;
        let y = xp[j];
        let x0 = xp[i];
        xp[i] = x0 + dx;
        self.balances[i] = xp[i];

        let price_scale = self.price_scale;

        // Normalize xp
        xp = [
            xp[0] * self.precisions[0],
            xp[1] * price_scale * self.precisions[1] / PRECISION,
        ];

        let prec_i = self.precisions[i];
        let prec_j = self.precisions[j];

        // Ramp handling
        let t = self.future_a_gamma_time;
        if t > U256::ZERO && U256::from(block_timestamp) < t {
            let mut x0_norm = x0 * prec_i;
            if i > 0 {
                x0_norm = x0_norm * price_scale / PRECISION;
            }
            let x1 = xp[i];
            xp[i] = x0_norm;
            self.d = if self.is_stable {
                crate::core::twocrypto_stable::get_d(&xp, a_gamma[0])?
            } else {
                twocrypto_ng::newton_d(a_gamma[0], a_gamma[1], xp, U256::ZERO)?
            };
            xp[i] = x1;
        }

        let d_current = self.d;
        let (y_out, k0_prev) = if self.is_stable {
            let y = crate::core::twocrypto_stable::get_y(i, j, xp[i], &xp, d_current, a_gamma[0])?;
            (y, U256::ZERO)
        } else {
            twocrypto_ng::get_y_2_ng(a_gamma[0], a_gamma[1], xp, d_current, j)?
        };

        let dy_internal = xp[j] - y_out;
        xp[j] = xp[j] - dy_internal;
        let dy_internal = dy_internal - U256::from(1u64);

        let mut dy = if j > 0 {
            dy_internal * PRECISION / price_scale
        } else {
            dy_internal
        };
        dy = dy / prec_j;

        let fee_amount = fee_2(&xp, self.mid_fee, self.out_fee, self.fee_gamma, self.is_stable) * dy
            / U256::from(10u64).pow(U256::from(10u64));
        dy = dy - fee_amount;

        let y_new = y - dy;
        self.balances[j] = y_new;

        let mut y_norm = y_new * prec_j;
        if j > 0 {
            y_norm = y_norm * price_scale / PRECISION;
        }
        xp[j] = y_norm;

        // On-chain computes D from post-swap xp BEFORE calling tweak_price,
        // then passes it as argument. Match that exactly.
        let d_new = if self.is_stable {
            crate::core::twocrypto_stable::get_d(&xp, a_gamma[0]).unwrap_or(U256::ZERO)
        } else {
            twocrypto_ng::newton_d(a_gamma[0], a_gamma[1], xp, k0_prev).unwrap_or(U256::ZERO)
        };
        self.tweak_price(a_gamma, xp, d_new, k0_prev, block_timestamp);

        eprintln!("  TwoCryptoNG exchange: dy={dy}, D={}, ps={}", self.d, self.price_scale);

        Some(ExchangeResult { dy, fee: fee_amount })
    }

    fn tweak_price(
        &mut self,
        a_gamma: [U256; 2],
        _xp: [U256; 2],
        new_d: U256,
        k0_prev: U256,
        block_timestamp: u64,
    ) {
        let mut price_oracle = self.price_oracle;
        let mut last_prices = self.last_prices;
        let price_scale = self.price_scale;

        let total_supply = self.total_supply;
        let old_xcp_profit = self.xcp_profit;
        let old_virtual_price = self.virtual_price;

        // On-chain reads `last_timestamp` into a local before updating. The
        // rebalance branch (below) is gated on the SAME local — meaning only
        // the first swap per block can trigger rebalancing.
        let last_timestamp_was_old = self.last_prices_timestamp < block_timestamp;

        // Update MA
        if last_timestamp_was_old {
            let ma_time = self.ma_time;
            let dt = U256::from(block_timestamp - self.last_prices_timestamp);
            let power = -alloy_primitives::I256::try_from(dt * WAD / ma_time).unwrap();
            let alpha = wad_exp(power);

            let capped = last_prices.min(U256::from(2u64) * price_scale);
            price_oracle = (capped * (WAD - alpha) + price_oracle * alpha) / WAD;

            self.price_oracle = price_oracle;
            self.last_prices_timestamp = block_timestamp;
        }

        eprintln!("  tweak_price_2ng: _xp=[{},{}], k0_prev={k0_prev}, is_stable={}", _xp[0], _xp[1], self.is_stable);
        // D
        let d_unadjusted = if new_d.is_zero() {
            if self.is_stable {
                crate::core::twocrypto_stable::get_d(&_xp, a_gamma[0]).unwrap_or(self.d)
            } else {
                twocrypto_ng::newton_d(a_gamma[0], a_gamma[1], _xp, k0_prev).unwrap_or(self.d)
            }
        } else {
            new_d
        };

        // last_prices via get_p
        let p = if self.is_stable {
            get_p_2_stable(_xp, d_unadjusted, a_gamma)
        } else {
            get_p_2(_xp, d_unadjusted, a_gamma)
        };
        self.last_prices = p * price_scale / WAD;

        // Virtual price
        let n = U256::from(N_COINS);
        let xp = [
            d_unadjusted / n,
            d_unadjusted * PRECISION / (n * price_scale),
        ];

        let mut xcp_profit = WAD;
        let mut virtual_price = WAD;

        if old_virtual_price > U256::ZERO {
            let xcp = isqrt(xp[0] * xp[1]);
            virtual_price = WAD * xcp / total_supply;
            // Vyper 0.4.3 (is_stable): additive xcp_profit update
            // Vyper 0.3.x (TwoCryptoNG): multiplicative xcp_profit update
            xcp_profit = if self.is_stable {
                old_xcp_profit + virtual_price - old_virtual_price
            } else {
                old_xcp_profit * virtual_price / old_virtual_price
            };

            if virtual_price < old_virtual_price && self.future_a_gamma_time < U256::from(block_timestamp) {
                return;
            }
        }
        self.xcp_profit = xcp_profit;

        // Rebalance — Vyper 0.4.3 (is_stable) pools gate rebalancing on
        // `last_timestamp < block.timestamp`: only the first swap per block
        // can trigger rebalancing. Vyper 0.3.10 TwoCryptoNG pools don't have
        // this guard and allow rebalancing on every swap.
        let rebalance_allowed = if self.is_stable { last_timestamp_was_old } else { true };
        if rebalance_allowed
            && virtual_price * U256::from(2u64) - WAD
                > xcp_profit + U256::from(2u64) * self.allowed_extra_profit
        {
            let mut norm = price_oracle * WAD / price_scale;
            norm = if norm > WAD { norm - WAD } else { WAD - norm };

            let adjustment_step = self.adjustment_step.max(norm / U256::from(5u64));

            if norm > adjustment_step && old_virtual_price > U256::ZERO {
                let p_new = (price_scale * (norm - adjustment_step)
                    + adjustment_step * price_oracle)
                    / norm;

                let xp_adj = [_xp[0], _xp[1] * p_new / price_scale];

                let d_opt = if self.is_stable {
                    crate::core::twocrypto_stable::get_d(&xp_adj, a_gamma[0])
                } else {
                    twocrypto_ng::newton_d(a_gamma[0], a_gamma[1], xp_adj, U256::ZERO)
                };

                if let Some(d) = d_opt {
                    let xp2 = [d / n, d * PRECISION / (n * p_new)];
                    let vp = WAD * isqrt(xp2[0] * xp2[1]) / total_supply;

                    if vp > WAD && (U256::from(2u64) * vp - WAD > xcp_profit) {
                        self.price_scale = p_new;
                        self.d = d;
                        self.virtual_price = vp;
                        return;
                    }
                }
            }
        }

        self.d = d_unadjusted;
        self.virtual_price = virtual_price;
    }
}
