//! Full exchange simulation for CryptoSwap pools.
//!
//! Replicates the on-chain `_exchange` + `tweak_price` logic exactly,
//! including D, price_scale, and price oracle updates.

use alloy_primitives::U256;

use crate::core::twocrypto_v1::{self, WAD, A_MULTIPLIER};

const N_COINS: U256 = U256::from_limbs([2, 0, 0, 0]);
const PRECISION: U256 = WAD;
const EXP_PRECISION: U256 = U256::from_limbs([10_000_000_000, 0, 0, 0]); // 10^10

/// Full mutable state for a TwoCryptoV1 pool, sufficient to simulate
/// exact on-chain exchange() behavior including tweak_price.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TwoCryptoV1State {
    pub balances: [U256; 2],
    pub price_scale: U256,
    pub price_oracle: U256,
    pub last_prices: U256,
    pub last_prices_timestamp: u64,
    pub d: U256,
    pub virtual_price: U256,
    pub xcp_profit: U256,
    /// Cached `xcp_profit` value at the previous `_claim_admin_fees` call.
    /// On-chain Vyper (twocrypto_v1.vy `_claim_admin_fees` L479, L511):
    ///     fees = (xcp_profit - xcp_profit_a) * admin_fee / (2 * 10**10)
    ///     ...
    ///     if xcp_profit > xcp_profit_a: self.xcp_profit_a = xcp_profit
    /// Required for wei-exact ClaimAdminFee replay because `fees` feeds
    /// into `frac = vprice * 1e18 / (vprice - fees) - 1e18` which
    /// determines the LP mint amount.
    pub xcp_profit_a: U256,
    /// Per-pool mutable admin fee (stored as `admin_fee: public(uint256)`).
    /// NG CryptoSwap pools have this as a compile-time constant at 5e9; V1
    /// pools make it a storage field set at construction and mutable via
    /// `apply_new_parameters()`. Canonical value is 5e9 (50% of the swap
    /// fee), but we read it at init to be robust against non-default
    /// deployments.
    pub admin_fee: U256,
    pub ann: U256,
    pub gamma: U256,
    pub mid_fee: U256,
    pub out_fee: U256,
    pub fee_gamma: U256,
    pub precisions: [U256; 2],
    pub total_supply: U256,
    pub allowed_extra_profit: U256,
    pub adjustment_step: U256,
    pub ma_half_time: U256,
    pub not_adjusted: bool,
    pub future_a_gamma_time: U256,
    /// Selects `newton_y_2`'s `mul2` formula per upstream fix for
    /// CurveCryptoSwap2ETH vs CurveCryptoSwap2. Factory pools (incl. all
    /// WETH-pair deployments) use `true`; legacy non-factory non-ETH pools
    /// use `false`. Defaults to `true` since Base-factory pools are always
    /// ETH variant.
    pub eth_variant: bool,
}

/// Result of a simulated exchange.
pub struct ExchangeResult {
    pub dy: U256,
    pub fee: U256,
}

// ── Math helpers ────────────────────────────────────────────────────────────

/// Geometric mean of 2 values via Newton iteration.
/// Port of Vyper `geometric_mean(x, sort)`.
pub fn geometric_mean(unsorted_x: [U256; 2], sort: bool) -> U256 {
    let x = if sort && unsorted_x[0] < unsorted_x[1] {
        [unsorted_x[1], unsorted_x[0]]
    } else {
        unsorted_x
    };
    let mut d = x[0];
    for _ in 0..255 {
        let d_prev = d;
        d = (d + x[0] * x[1] / d) / U256::from(2u64);
        let diff = if d > d_prev { d - d_prev } else { d_prev - d };
        if diff <= U256::from(1u64) || diff * WAD < d {
            return d;
        }
    }
    d
}

/// 1e18 * 0.5^(power/1e18)
/// Port of Vyper `halfpow`.
pub fn halfpow(power: U256) -> U256 {
    let intpow = power / WAD;
    let otherpow = power - intpow * WAD;
    if intpow > U256::from(59u64) {
        return U256::ZERO;
    }
    let result = WAD / (U256::from(1u64) << intpow.as_limbs()[0] as usize);
    if otherpow.is_zero() {
        return result;
    }

    let mut term = WAD;
    let x = U256::from(500_000_000_000_000_000u64); // 5e17
    let mut s = WAD;
    let mut neg = false;

    for i in 1u64..256 {
        let k = U256::from(i) * WAD;
        let c_val = k - WAD;
        let c;
        if otherpow > c_val {
            c = otherpow - c_val;
            neg = !neg;
        } else {
            c = c_val - otherpow;
        }
        term = term * (c * x / WAD) / k;
        if neg {
            s = s - term;
        } else {
            s = s + term;
        }
        if term < EXP_PRECISION {
            return result * s / WAD;
        }
    }
    result * s / WAD
}

/// Fee calculation. Port of Vyper `_fee(xp)`.
fn fee(xp: &[U256; 2], mid_fee: U256, out_fee: U256, fee_gamma: U256) -> U256 {
    let f_sum = xp[0] + xp[1];
    let f = fee_gamma * WAD
        / (fee_gamma + WAD - WAD * U256::from(4u64) * xp[0] / f_sum * xp[1] / f_sum);
    (mid_fee * f + out_fee * (WAD - f)) / WAD
}

impl TwoCryptoV1State {
    /// Simulate exchange(i, j, dx) and return the output amount.
    /// Mutates all internal state exactly as the on-chain contract would.
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

        xp = [
            xp[0] * self.precisions[0],
            xp[1] * price_scale * self.precisions[1] / PRECISION,
        ];

        let prec_i = self.precisions[i];
        let prec_j = self.precisions[j];

        // In case ramp is happening
        let t = self.future_a_gamma_time;
        if t > U256::ZERO {
            let mut x0_norm = x0 * prec_i;
            if i > 0 {
                x0_norm = x0_norm * price_scale / PRECISION;
            }
            let x1 = xp[i];
            xp[i] = x0_norm;
            self.d = twocrypto_v1::newton_d(a_gamma[0], a_gamma[1], xp)?;
            xp[i] = x1;
            if U256::from(block_timestamp) >= t {
                self.future_a_gamma_time = U256::from(1u64);
            }
        }

        let d_current = self.d;
        let dy_internal = xp[j]
            - twocrypto_v1::newton_y_2(
                a_gamma[0], a_gamma[1], xp, d_current, j, self.eth_variant,
            )?;
        xp[j] = xp[j] - dy_internal;
        let dy_internal = dy_internal - U256::from(1u64);

        let mut dy = if j > 0 {
            dy_internal * PRECISION / price_scale
        } else {
            dy_internal
        };
        dy = dy / prec_j;

        let fee_amount = fee(&xp, self.mid_fee, self.out_fee, self.fee_gamma) * dy
            / U256::from(10u64).pow(U256::from(10u64));
        dy = dy - fee_amount;

        let y_new = y - dy;
        self.balances[j] = y_new;

        let mut y_norm = y_new * prec_j;
        if j > 0 {
            y_norm = y_norm * price_scale / PRECISION;
        }
        xp[j] = y_norm;

        // Calculate price for oracle
        let p = if dx > U256::from(100_000u64) && dy > U256::from(100_000u64) {
            let _dx = dx * prec_i;
            let _dy = dy * prec_j;
            if i == 0 {
                _dx * WAD / _dy
            } else {
                _dy * WAD / _dx
            }
        } else {
            U256::ZERO
        };

        self.tweak_price(a_gamma, xp, p, U256::ZERO, block_timestamp);

        Some(ExchangeResult { dy, fee: fee_amount })
    }

    /// Apply add_liquidity: update balances, compute D, run tweak_price.
    /// `amounts` are the token amounts added (in native units).
    /// Returns the new D.
    pub fn apply_add_liquidity(&mut self, amounts: [U256; 2], block_timestamp: u64) -> Option<U256> {
        let precisions = self.precisions;
        let price_scale = self.price_scale;

        // Update balances
        self.balances[0] = self.balances[0] + amounts[0];
        self.balances[1] = self.balances[1] + amounts[1];

        // Compute xp
        let xp = [
            self.balances[0] * precisions[0],
            self.balances[1] * price_scale * precisions[1] / PRECISION,
        ];

        // Compute new D
        let d = twocrypto_v1::newton_d(self.ann, self.gamma, xp)?;

        // Run tweak_price with the new D (same as on-chain add_liquidity)
        self.tweak_price([self.ann, self.gamma], xp, U256::ZERO, d, block_timestamp);

        Some(self.d)
    }

    /// Apply remove_liquidity_one_coin: replicate V1 _calc_withdraw_one_coin + tweak_price.
    pub fn apply_remove_liquidity_one(
        &mut self,
        i: usize,
        token_amount: U256,
        coin_amount: U256,
        block_timestamp: u64,
    ) -> Option<U256> {
        let precisions = self.precisions;
        let price_scale = self.price_scale;

        let xx = self.balances;
        let price_scale_i_full = price_scale * precisions[1]; // for coin 1
        let mut xp = [
            xx[0] * precisions[0],
            xx[1] * price_scale_i_full / PRECISION,
        ];
        let price_scale_i = if i == 0 {
            PRECISION * precisions[0]
        } else {
            price_scale_i_full
        };

        // _calc_withdraw_one_coin logic:
        let d0 = self.d;
        let fee_rate = fee(&xp, self.mid_fee, self.out_fee, self.fee_gamma);
        let dd = token_amount * d0 / self.total_supply;
        // D adjusted for fee: D -= (dD - (fee * dD / (2 * 1e10) + 1))
        let d_adj = d0 - (dd - (fee_rate * dd / (U256::from(2u64) * U256::from(10u64).pow(U256::from(10u64))) + U256::from(1u64)));

        let y = twocrypto_v1::newton_y_2(
            self.ann, self.gamma, xp, d_adj, i, self.eth_variant,
        )?;
        xp[i] = y;

        // Compute price for oracle
        let dy = coin_amount; // from event
        let mut p = U256::ZERO;
        if dy > U256::from(100_000u64) && token_amount > U256::from(100_000u64) {
            let s = if i == 1 {
                xx[0] * precisions[0]
            } else {
                xx[1] * precisions[1]
            };
            let s_dd = s * dd / d0;
            let precision_i = precisions[i];
            let denom = dy * precision_i - dd * xx[i] * precision_i / d0;
            if !denom.is_zero() {
                p = s_dd * PRECISION / denom;
                if i == 0 {
                    p = PRECISION * PRECISION / p;
                }
            }
        }

        // Update balance and burn tokens BEFORE tweak_price
        // (on-chain burns tokens before calling tweak_price, so
        // total_supply used in virtual_price computation is post-burn)
        self.balances[i] = self.balances[i] - coin_amount;
        self.total_supply = self.total_supply - token_amount;

        // tweak_price with the fee-adjusted D
        self.tweak_price([self.ann, self.gamma], xp, p, d_adj, block_timestamp);

        Some(self.d)
    }

    /// Port of Vyper `tweak_price`.
    fn tweak_price(
        &mut self,
        a_gamma: [U256; 2],
        _xp: [U256; 2],
        p_i: U256,
        new_d: U256,
        block_timestamp: u64,
    ) {
        let mut price_oracle = self.price_oracle;
        let mut last_prices = self.last_prices;
        let price_scale = self.price_scale;
        let last_prices_timestamp = self.last_prices_timestamp;

        if last_prices_timestamp < block_timestamp {
            let ma_half_time = self.ma_half_time;
            let alpha = halfpow(
                U256::from(block_timestamp - last_prices_timestamp) * WAD / ma_half_time,
            );
            price_oracle = (last_prices * (WAD - alpha) + price_oracle * alpha) / WAD;
            self.price_oracle = price_oracle;
            self.last_prices_timestamp = block_timestamp;
        }

        let d_unadjusted = if new_d.is_zero() {
            twocrypto_v1::newton_d(a_gamma[0], a_gamma[1], _xp)
                .unwrap_or(self.d)
        } else {
            new_d
        };

        if p_i > U256::ZERO {
            last_prices = p_i;
        } else {
            // Calculate real prices
            let mut __xp = _xp;
            let dx_price = __xp[0] / U256::from(1_000_000u64);
            __xp[0] = __xp[0] + dx_price;
            let y_out = twocrypto_v1::newton_y_2(
                a_gamma[0], a_gamma[1], __xp, d_unadjusted, 1, self.eth_variant,
            );
            if let Some(y) = y_out {
                if _xp[1] > y {
                    last_prices = price_scale * dx_price / (_xp[1] - y);
                }
            }
        }
        self.last_prices = last_prices;

        let total_supply = self.total_supply;
        let old_xcp_profit = self.xcp_profit;
        let old_virtual_price = self.virtual_price;

        let xp = [
            d_unadjusted / U256::from(2u64),
            d_unadjusted * PRECISION / (U256::from(2u64) * price_scale),
        ];

        let mut xcp_profit = WAD;
        let mut virtual_price = WAD;

        if old_virtual_price > U256::ZERO {
            let xcp = geometric_mean(xp, true);
            virtual_price = WAD * xcp / total_supply;
            xcp_profit = old_xcp_profit * virtual_price / old_virtual_price;

            let t = self.future_a_gamma_time;
            if virtual_price < old_virtual_price && t.is_zero() {
                // "Loss" — on-chain would revert, we just keep old values
                return;
            }
            if t == U256::from(1u64) {
                self.future_a_gamma_time = U256::ZERO;
            }
        }
        self.xcp_profit = xcp_profit;

        let mut norm = price_oracle * WAD / price_scale;
        if norm > WAD {
            norm = norm - WAD;
        } else {
            norm = WAD - norm;
        }
        let adjustment_step = self.adjustment_step.max(norm / U256::from(5u64));

        let mut needs_adjustment = self.not_adjusted;
        if !needs_adjustment
            && (virtual_price * U256::from(2u64) - WAD
                > xcp_profit + U256::from(2u64) * self.allowed_extra_profit)
            && (norm > adjustment_step)
            && (old_virtual_price > U256::ZERO)
        {
            needs_adjustment = true;
            self.not_adjusted = true;
        }

        if needs_adjustment {
            if norm > adjustment_step && old_virtual_price > U256::ZERO {
                let p_new = (price_scale * (norm - adjustment_step)
                    + adjustment_step * price_oracle)
                    / norm;

                let xp_adj = [_xp[0], _xp[1] * p_new / price_scale];
                let d_adj = twocrypto_v1::newton_d(
                    a_gamma[0],
                    a_gamma[1],
                    xp_adj,
                );
                if let Some(d) = d_adj {
                    let xp2 = [
                        d / U256::from(2u64),
                        d * PRECISION / (U256::from(2u64) * p_new),
                    ];
                    let vp = WAD * geometric_mean(xp2, true) / total_supply;

                    if vp > WAD && (U256::from(2u64) * vp - WAD > xcp_profit) {
                        self.price_scale = p_new;
                        self.d = d;
                        self.virtual_price = vp;
                        return;
                    }
                }

                self.not_adjusted = false;
                self.d = d_unadjusted;
                self.virtual_price = virtual_price;
                self.claim_admin_fees();
                return;
            }
        }

        self.d = d_unadjusted;
        self.virtual_price = virtual_price;

        if needs_adjustment {
            self.not_adjusted = false;
            self.claim_admin_fees();
        }
    }

    /// Port of Vyper `_claim_admin_fees()` (twocrypto_v1.vy L475-511).
    ///
    /// Called from `tweak_price` in non-adjustment paths. On-chain this:
    ///   1. Gulps balances (no-op for non-rebasing tokens — skipped here)
    ///   2. If xcp_profit > xcp_profit_a and fees > 0 and receiver != 0:
    ///      mints admin LP tokens, decrements xcp_profit
    ///   3. Recalculates D via newton_D(xp)
    ///   4. Updates virtual_price = 1e18 * get_xcp(D) / total_supply
    ///   5. If xcp_profit > xcp_profit_a: sets xcp_profit_a = xcp_profit
    pub fn claim_admin_fees(&mut self) {
        let xcp_profit = self.xcp_profit;
        let xcp_profit_a = self.xcp_profit_a;
        // Skip gulp — our balances are event-tracked
        let vprice = self.virtual_price;

        if xcp_profit > xcp_profit_a {
            let two_fee_denom = U256::from(2u64) * EXP_PRECISION; // 2 * 10^10
            let fees = (xcp_profit - xcp_profit_a) * self.admin_fee / two_fee_denom;
            if !fees.is_zero() && vprice > fees {
                // frac = vprice * 1e18 / (vprice - fees) - 1e18
                let frac = vprice * WAD / (vprice - fees) - WAD;
                // claimed = total_supply * frac / 1e18 (mint_relative)
                let claimed = self.total_supply * frac / WAD;
                self.xcp_profit = xcp_profit - fees * U256::from(2u64);
                self.total_supply = self.total_supply + claimed;
            }
        }

        // Recalculate D from current balances + price_scale
        let xp = [
            self.balances[0] * self.precisions[0],
            self.balances[1] * self.price_scale * self.precisions[1] / PRECISION,
        ];
        if let Some(new_d) = twocrypto_v1::newton_d(self.ann, self.gamma, xp) {
            self.d = new_d;
            // virtual_price = 1e18 * get_xcp(D) / total_supply
            // get_xcp(D) = geometric_mean([D/2, D*1e18/(2*price_scale)])
            let x = [
                new_d / N_COINS,
                new_d * PRECISION / (N_COINS * self.price_scale),
            ];
            let xcp = geometric_mean(x, true);
            if !self.total_supply.is_zero() {
                self.virtual_price = WAD * xcp / self.total_supply;
            }
        }

        // "if xcp_profit > xcp_profit_a: self.xcp_profit_a = xcp_profit"
        // Uses original xcp_profit_a (local var) and possibly-decremented xcp_profit
        if self.xcp_profit > xcp_profit_a {
            self.xcp_profit_a = self.xcp_profit;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halfpow_120e15() {
        let power = U256::from(120_000_000_000_000_000u128);
        let result = halfpow(power);
        // On-chain verified value from TwoCryptoV1 pool 0x11c1 on Base
        let expected = U256::from(920187657650015204u128);
        assert_eq!(result, expected, "halfpow(120e15) mismatch: got {result}, expected {expected}");
    }
}
