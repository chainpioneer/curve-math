//! Full exchange simulation for TriCryptoV1 pools.
//!
//! Replicates on-chain `_exchange` + `tweak_price` exactly.
//! Uses `halfpow` for oracle EMA (same as TwoCryptoV1). 3 coins, 2 price_scales.
//! Only 2 pools ever deployed (tricrypto2 on Mainnet), but needed for completeness.

use alloy_primitives::U256;

use crate::core::tricrypto_v1::{self, WAD, FEE_DENOMINATOR, A_MULTIPLIER};
use crate::core::twocrypto_ng::isqrt;
use crate::exchange::{halfpow, geometric_mean};

const N_COINS: u64 = 3;
const PRECISION: U256 = WAD;

/// Full mutable state for a TriCryptoV1 pool.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TriCryptoV1State {
    pub balances: [U256; 3],
    pub price_scale: [U256; 2],
    pub price_oracle: [U256; 2],
    pub last_prices: [U256; 2],
    pub last_prices_timestamp: u64,
    pub d: U256,
    pub virtual_price: U256,
    pub xcp_profit: U256,
    /// Cached `xcp_profit` value at the previous `_claim_admin_fees` call.
    /// Same semantics as `TwoCryptoV1State::xcp_profit_a`. Required for
    /// wei-exact ClaimAdminFee replay on the LP-mint flavor: `fees =
    /// (xcp_profit - xcp_profit_a) * admin_fee / 2e10` drives `frac`
    /// which drives the LP mint amount.
    pub xcp_profit_a: U256,
    /// Per-pool mutable admin fee. Same semantics as
    /// `TwoCryptoV1State::admin_fee`. Set at construction (canonical 5e9)
    /// and mutable via admin actions.
    pub admin_fee: U256,
    pub ann: U256,
    pub gamma: U256,
    pub mid_fee: U256,
    pub out_fee: U256,
    pub fee_gamma: U256,
    pub precisions: [U256; 3],
    pub total_supply: U256,
    pub allowed_extra_profit: U256,
    pub adjustment_step: U256,
    pub ma_half_time: U256,
    pub not_adjusted: bool,
    pub future_a_gamma_time: U256,
}

pub struct ExchangeResult {
    pub dy: U256,
    pub fee: U256,
}

/// Fee for 3-coin pool (same formula as TwoCryptoV1 but 3 coins).
fn fee_3(xp: &[U256; 3], mid_fee: U256, out_fee: U256, fee_gamma: U256) -> U256 {
    let s = xp[0] + xp[1] + xp[2];
    let n = U256::from(N_COINS);
    // V1 formula: K = WAD * N^N, then K = K * x_i / S for each coin
    let mut nn = U256::from(1u64);
    for _ in 0..3 { nn = nn * n; }
    let mut k = WAD * nn;
    for x_i in xp {
        k = k * (*x_i) / s;
    }
    let f = if fee_gamma > U256::ZERO {
        fee_gamma * WAD / (fee_gamma + WAD - k)
    } else {
        k
    };
    (mid_fee * f + out_fee * (WAD - f)) / WAD
}

/// Geometric mean of 3 values using Newton iteration (V1 style).
pub fn geometric_mean_3(x: [U256; 3]) -> U256 {
    // Sort descending for better convergence
    let mut sorted = x;
    if sorted[0] < sorted[1] { sorted.swap(0, 1); }
    if sorted[1] < sorted[2] { sorted.swap(1, 2); }
    if sorted[0] < sorted[1] { sorted.swap(0, 1); }

    let mut d = sorted[0];
    let n = U256::from(3u64);
    for _ in 0..255u32 {
        let d_prev = d;
        // D = (D + prod(x) / D^(N-1)) / N
        // For 3 coins: D = (2*D + x0*x1*x2 / D^2) / 3
        let prod_over_d2 = sorted[0] * sorted[1] / d * sorted[2] / d;
        d = (U256::from(2u64) * d + prod_over_d2) / n;
        let diff = if d > d_prev { d - d_prev } else { d_prev - d };
        if diff <= U256::from(1u64) || diff * WAD < d {
            return d;
        }
    }
    d
}

/// V1 newton_D for 3 coins (uses geometric_mean_3 for initial guess).
fn newton_d_3_v1(ann: U256, gamma: U256, x_unsorted: [U256; 3]) -> Option<U256> {
    let mut x = x_unsorted;
    // Sort descending
    if x[0] < x[1] { x.swap(0, 1); }
    if x[1] < x[2] { x.swap(1, 2); }
    if x[0] < x[1] { x.swap(0, 1); }

    if x[0].is_zero() { return None; }

    let n = U256::from(N_COINS);
    let s = x[0] + x[1] + x[2];
    let mut d = n * geometric_mean_3(x);
    let g1k0_base = gamma + WAD;

    for _ in 0..255u32 {
        let d_prev = d;
        if d.is_zero() { return None; }

        // K0 = 10^18 * x[0] * N / D * x[1] * N / D * x[2] * N / D
        let k0 = WAD * x[0] * n / d * x[1] * n / d * x[2] * n / d;

        let _g1k0 = if g1k0_base > k0 {
            g1k0_base - k0 + U256::from(1u64)
        } else {
            k0 - g1k0_base + U256::from(1u64)
        };

        let mul1 = WAD * d / gamma * _g1k0 / gamma * _g1k0 * A_MULTIPLIER / ann;
        let mul2 = U256::from(2u64) * WAD * n * k0 / _g1k0;

        if k0.is_zero() { return None; }
        let neg_fprime = (s + s * mul2 / WAD) + mul1 * n / k0 - mul2 * d / WAD;
        if neg_fprime.is_zero() { return None; }

        let d_plus = d * (neg_fprime + s) / neg_fprime;
        let mut d_minus = d * d / neg_fprime;

        if WAD > k0 {
            d_minus += d * (mul1 / neg_fprime) / WAD * (WAD - k0) / k0;
        } else {
            d_minus -= d * (mul1 / neg_fprime) / WAD * (k0 - WAD) / k0;
        }

        d = if d_plus > d_minus { d_plus - d_minus } else { (d_minus - d_plus) / U256::from(2u64) };

        let diff = if d > d_prev { d - d_prev } else { d_prev - d };
        let threshold = U256::from(10u64).pow(U256::from(16u64)).max(d);
        if diff * U256::from(10u64).pow(U256::from(14u64)) < threshold {
            return Some(d);
        }
    }
    None
}

impl TriCryptoV1State {
    pub fn exchange(&mut self, i: usize, j: usize, dx: U256, block_timestamp: u64) -> Option<ExchangeResult> {
        if i == j || i >= 3 || j >= 3 || dx.is_zero() { return None; }

        let a_gamma = [self.ann, self.gamma];
        let mut xp = self.balances;
        let y = xp[j];
        let x0 = xp[i];
        xp[i] = x0 + dx;
        self.balances[i] = xp[i];

        let price_scale = self.price_scale;

        // Normalize
        xp[0] = xp[0] * self.precisions[0];
        for k in 1..3 {
            xp[k] = xp[k] * price_scale[k - 1] * self.precisions[k] / PRECISION;
        }

        let prec_i = self.precisions[i];
        let prec_j = self.precisions[j];

        // Ramp
        let t = self.future_a_gamma_time;
        if t > U256::ZERO {
            let mut x0_norm = x0 * prec_i;
            if i > 0 { x0_norm = x0_norm * price_scale[i - 1] / PRECISION; }
            let x1 = xp[i];
            xp[i] = x0_norm;
            self.d = newton_d_3_v1(a_gamma[0], a_gamma[1], xp)?;
            xp[i] = x1;
            if U256::from(block_timestamp) >= t {
                self.future_a_gamma_time = U256::from(1u64);
            }
        }

        let d_current = self.d;
        let dy_internal = xp[j] - tricrypto_v1::newton_y_3(a_gamma[0], a_gamma[1], xp, d_current, j)?;
        xp[j] = xp[j] - dy_internal;
        let dy_internal = dy_internal - U256::from(1u64);

        let mut dy = if j > 0 { dy_internal * PRECISION / price_scale[j - 1] } else { dy_internal };
        dy = dy / prec_j;

        let fee_amount = fee_3(&xp, self.mid_fee, self.out_fee, self.fee_gamma) * dy
            / U256::from(10u64).pow(U256::from(10u64));
        dy = dy - fee_amount;

        let y_new = y - dy;
        self.balances[j] = y_new;

        let mut y_norm = y_new * prec_j;
        if j > 0 { y_norm = y_norm * price_scale[j - 1] / PRECISION; }
        xp[j] = y_norm;

        // V1: compute price p inline
        let p = if dx > U256::from(100_000u64) && dy > U256::from(100_000u64) {
            let _dx = dx * prec_i;
            let _dy = dy * prec_j;
            if i == 0 { _dx * WAD / _dy } else { _dy * WAD / _dx }
        } else {
            U256::ZERO
        };

        self.tweak_price(a_gamma, xp, p, U256::ZERO, block_timestamp);

        Some(ExchangeResult { dy, fee: fee_amount })
    }

    fn tweak_price(
        &mut self,
        a_gamma: [U256; 2],
        _xp: [U256; 3],
        p_i: U256,
        new_d: U256,
        block_timestamp: u64,
    ) {
        let mut price_oracle = self.price_oracle;
        let mut last_prices = self.last_prices;
        let price_scale = self.price_scale;

        let total_supply = self.total_supply;
        let old_xcp_profit = self.xcp_profit;
        let old_virtual_price = self.virtual_price;

        // V1 uses halfpow for EMA
        if self.last_prices_timestamp < block_timestamp {
            let ma_half_time = self.ma_half_time;
            let alpha = halfpow(
                U256::from(block_timestamp - self.last_prices_timestamp) * WAD / ma_half_time,
            );
            for k in 0..2 {
                price_oracle[k] = (last_prices[k] * (WAD - alpha) + price_oracle[k] * alpha) / WAD;
            }
            self.price_oracle = price_oracle;
            self.last_prices_timestamp = block_timestamp;
        }

        let d_unadjusted = if new_d.is_zero() {
            newton_d_3_v1(a_gamma[0], a_gamma[1], _xp).unwrap_or(self.d)
        } else {
            new_d
        };

        // V1: p_i is for a single pair. For 3-coin, compute prices numerically.
        if p_i > U256::ZERO {
            // p_i applies to the pair that was swapped
            // For simplicity, recompute both prices numerically
            for k in 0..2 {
                let mut xp_mod = _xp;
                let dx_price = xp_mod[0] / U256::from(1_000_000u64);
                xp_mod[0] = xp_mod[0] + dx_price;
                if let Some(y) = tricrypto_v1::newton_y_3(a_gamma[0], a_gamma[1], xp_mod, d_unadjusted, k + 1) {
                    if _xp[k + 1] > y {
                        last_prices[k] = price_scale[k] * dx_price / (_xp[k + 1] - y);
                    }
                }
            }
        } else {
            for k in 0..2 {
                let mut xp_mod = _xp;
                let dx_price = xp_mod[0] / U256::from(1_000_000u64);
                xp_mod[0] = xp_mod[0] + dx_price;
                if let Some(y) = tricrypto_v1::newton_y_3(a_gamma[0], a_gamma[1], xp_mod, d_unadjusted, k + 1) {
                    if _xp[k + 1] > y {
                        last_prices[k] = price_scale[k] * dx_price / (_xp[k + 1] - y);
                    }
                }
            }
        }
        self.last_prices = last_prices;

        let n = U256::from(N_COINS);
        let mut xp = [U256::ZERO; 3];
        xp[0] = d_unadjusted / n;
        for k in 0..2 {
            xp[k + 1] = d_unadjusted * PRECISION / (n * price_scale[k]);
        }

        let mut xcp_profit = WAD;
        let mut virtual_price = WAD;

        if old_virtual_price > U256::ZERO {
            let xcp = geometric_mean_3(xp);
            virtual_price = WAD * xcp / total_supply;
            xcp_profit = old_xcp_profit * virtual_price / old_virtual_price;

            let t = self.future_a_gamma_time;
            if virtual_price < old_virtual_price && t.is_zero() {
                return;
            }
            if t == U256::from(1u64) {
                self.future_a_gamma_time = U256::ZERO;
            }
        }
        self.xcp_profit = xcp_profit;

        // Norm: L2 for 3-coin V1 (same as NG)
        let mut norm = U256::ZERO;
        for k in 0..2 {
            let ratio = price_oracle[k] * WAD / price_scale[k];
            let r = if ratio > WAD { ratio - WAD } else { WAD - ratio };
            norm = norm + r * r;
        }
        norm = isqrt(norm);

        let adjustment_step = self.adjustment_step.max(norm / U256::from(5u64));
        let mut needs_adjustment = self.not_adjusted;

        if !needs_adjustment
            && (virtual_price * U256::from(2u64) - WAD > xcp_profit + U256::from(2u64) * self.allowed_extra_profit)
            && (norm > adjustment_step)
            && (old_virtual_price > U256::ZERO)
        {
            needs_adjustment = true;
            self.not_adjusted = true;
        }

        if needs_adjustment {
            if norm > adjustment_step && old_virtual_price > U256::ZERO {
                let mut p_new = [U256::ZERO; 2];
                for k in 0..2 {
                    p_new[k] = (price_scale[k] * (norm - adjustment_step) + adjustment_step * price_oracle[k]) / norm;
                }

                let mut xp_adj = _xp;
                for k in 0..2 {
                    xp_adj[k + 1] = _xp[k + 1] * p_new[k] / price_scale[k];
                }

                if let Some(d) = newton_d_3_v1(a_gamma[0], a_gamma[1], xp_adj) {
                    let mut xp2 = [U256::ZERO; 3];
                    xp2[0] = d / n;
                    for k in 0..2 {
                        xp2[k + 1] = d * PRECISION / (n * p_new[k]);
                    }
                    let vp = WAD * geometric_mean_3(xp2) / total_supply;

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

    /// Port of Vyper `_claim_admin_fees()` for 3-coin V1 pools.
    /// Same logic as `TwoCryptoV1State::claim_admin_fees` but with 3 coins.
    /// Source: tricrypto_v1.vy (same pattern as twocrypto_v1.vy L475-511).
    pub fn claim_admin_fees(&mut self) {
        let xcp_profit = self.xcp_profit;
        let xcp_profit_a = self.xcp_profit_a;
        let vprice = self.virtual_price;
        let exp_precision = U256::from(10_000_000_000u64); // 10^10

        if xcp_profit > xcp_profit_a {
            let two_fee_denom = U256::from(2u64) * exp_precision;
            let fees = (xcp_profit - xcp_profit_a) * self.admin_fee / two_fee_denom;
            if !fees.is_zero() && vprice > fees {
                let frac = vprice * WAD / (vprice - fees) - WAD;
                let claimed = self.total_supply * frac / WAD;
                self.xcp_profit = xcp_profit - fees * U256::from(2u64);
                self.total_supply = self.total_supply + claimed;
            }
        }

        // Recalculate D from current balances + price_scale
        let n = U256::from(N_COINS);
        let xp = [
            self.balances[0] * self.precisions[0],
            self.balances[1] * self.price_scale[0] * self.precisions[1] / PRECISION,
            self.balances[2] * self.price_scale[1] * self.precisions[2] / PRECISION,
        ];
        if let Some(new_d) = newton_d_3_v1(self.ann, self.gamma, xp) {
            self.d = new_d;
            // get_xcp(D) for 3 coins: geometric_mean([D/N, D*1e18/(N*ps[0]), D*1e18/(N*ps[1])])
            let x = [
                new_d / n,
                new_d * PRECISION / (n * self.price_scale[0]),
                new_d * PRECISION / (n * self.price_scale[1]),
            ];
            let xcp = geometric_mean_3(x);
            if !self.total_supply.is_zero() {
                self.virtual_price = WAD * xcp / self.total_supply;
            }
        }

        if self.xcp_profit > xcp_profit_a {
            self.xcp_profit_a = self.xcp_profit;
        }
    }
}
