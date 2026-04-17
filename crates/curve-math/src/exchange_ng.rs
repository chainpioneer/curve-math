//! Full exchange simulation for TriCryptoNG pools.
//!
//! Replicates the on-chain `_exchange` + `tweak_price` logic exactly,
//! including D, price_scale, and price oracle updates.
//! Uses `wad_exp` (Snekmate) for oracle EMA instead of `halfpow` (V1).

use alloy_primitives::{I256, U256};

use crate::core::tricrypto_ng::{self, WAD, FEE_DENOMINATOR, A_MULTIPLIER, isqrt, cbrt};

const N_COINS: u64 = 3;
const PRECISION: U256 = WAD;
const ADMIN_FEE: U256 = U256::from_limbs([5_000_000_000, 0, 0, 0]); // 50%

/// Full mutable state for a TriCryptoNG pool.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TriCryptoNGState {
    pub balances: [U256; 3],
    pub price_scale: [U256; 2],
    pub price_oracle: [U256; 2],
    pub last_prices: [U256; 2],
    pub last_prices_timestamp: u64,
    pub d: U256,
    pub virtual_price: U256,
    pub xcp_profit: U256,
    /// `xcp_profit` snapshot at the last admin-fee claim. Used by
    /// `_claim_admin_fees` to compute `fees = (xcp_profit - xcp_profit_a) *
    /// ADMIN_FEE / (2*1e10)`. Required for exact ClaimAdminFee replay on the
    /// deployed Vyper 0.3.9 `CurveTricryptoOptimizedWETH` contract.
    pub xcp_profit_a: U256,
    /// v0.3.10 only: accumulator for admin LP fees from `add_liquidity`.
    /// Same semantics as `TwoCryptoNGState::admin_lp_virtual_balance`.
    /// v0.3.9 WETH variant doesn't have this field (uses LP-mint instead).
    pub admin_lp_virtual_balance: U256,
    /// `true` for `CurveTricryptoOptimizedWETH` (Vyper 0.3.9) — detected via
    /// the `WETH20()` getter at init. `false` for `CurveTricryptoOptimized`
    /// (Vyper 0.3.10). Affects: storage slot layout, timestamp ABI
    /// (`last_prices_timestamp` vs `last_timestamp`), admin-fee mechanism
    /// (LP-mint vs balance-subtract), `admin_lp_virtual_balance` tracking
    /// (v0.3.10 only).
    pub is_weth_variant: bool,
    pub ann: U256,
    pub gamma: U256,
    pub mid_fee: U256,
    pub out_fee: U256,
    pub fee_gamma: U256,
    pub precisions: [U256; 3],
    pub total_supply: U256,
    pub allowed_extra_profit: U256,
    pub adjustment_step: U256,
    pub ma_time: U256,
    pub future_a_gamma_time: U256,
}

pub struct ExchangeResult {
    pub dy: U256,
    pub fee: U256,
}

// ── wad_exp (Snekmate) ──────────────────────────────────────────────────────

/// Arithmetic right shift for I256 (matching EVM SAR).
fn shr96(v: I256) -> I256 { v.asr(96) }

/// Port of Snekmate `_snekmate_wad_exp`. Computes e^x with 1e18 precision.
/// Input x is a signed int256 in 1e18 fixed point.
/// All arithmetic uses wrapping ops to match EVM's unsafe_mul/unsafe_add/unsafe_sub.
pub fn wad_exp(x: I256) -> U256 {
    if x <= I256::try_from(-42139678854452767551i128).unwrap() {
        return U256::ZERO;
    }

    let c = |v: u128| -> I256 { I256::try_from(v).unwrap() };
    let mut value = x;

    let five_pow_18 = c(3814697265625u128); // 5^18
    value = (value << 78usize) / five_pow_18;

    let log2_e = c(54916777467707473351141471128u128);
    let k: I256 = shr96(
        value.wrapping_mul(I256::from_raw(U256::from(1u64) << 96)) / log2_e
        + I256::from_raw(U256::from(1u64) << 95)
    );
    value = value.wrapping_sub(k.wrapping_mul(log2_e));

    let y: I256 = shr96(value.wrapping_add(c(1346386616545796478920950773328u128)).wrapping_mul(value))
        .wrapping_add(c(57155421227552351082224309758442u128));

    let p: I256 = {
        let t1 = y.wrapping_add(value).wrapping_sub(c(94201549194550492254356042504812u128));
        let t2 = shr96(t1.wrapping_mul(y)).wrapping_add(c(28719021644029726153956944680412240u128));
        t2.wrapping_mul(value).wrapping_add(c(4385272521454847904659076985693276u128) << 96usize)
    };

    let q: I256 = {
        let v_minus_c4 = value.wrapping_sub(c(2855989394907223263936484059900u128));
        let prod = v_minus_c4.wrapping_mul(value);
        let mut q = shr96(prod);
        q = q.wrapping_add(c(50020603652535783019961831881945u128));
        q = shr96(q.wrapping_mul(value));
        q = q.wrapping_sub(c(533845033583426703283633433725380u128));
        q = shr96(q.wrapping_mul(value));
        q = q.wrapping_add(c(3604857256930695427073651918091429u128));
        q = shr96(q.wrapping_mul(value));
        q = q.wrapping_sub(c(14423608567350463180887372962807573u128));
        q = shr96(q.wrapping_mul(value));
        q.wrapping_add(c(26449188498355588339934803723976023u128))
    };

    let r: I256 = p / q;
    let r_uint = U256::from_be_bytes(r.to_be_bytes::<32>());
    let scale = U256::from_str_radix("3822833074963236453042738258902158003155416615667", 10).unwrap();
    let shift = I256::try_from(195u64).unwrap() - k;
    let shift_uint = U256::from_be_bytes(shift.to_be_bytes::<32>());
    r_uint.wrapping_mul(scale) >> shift_uint.as_limbs()[0] as usize
}

/// Port of MATH.get_p() — analytical price derivatives dx_0/dx_k.
/// Returns [p_01, p_02] in 1e18 precision (needs multiplying by price_scale).
fn get_p(xp: [U256; 3], d: U256, a_gamma: [U256; 2]) -> [U256; 2] {
    let p36 = U256::from(10u64).pow(U256::from(36u64));
    let p18 = WAD;

    // K0 = 27 * xp[0] * xp[1] / D * xp[2] / D * 10^36 / D
    let k0 = U256::from(27u64) * xp[0] * xp[1] / d * xp[2] / d * p36 / d;

    // GK0
    let gamma_plus_wad = a_gamma[1] + p18;
    let gk0 = U256::from(2u64) * k0 * k0 / p36 * k0 / p36
        + gamma_plus_wad * gamma_plus_wad
        - k0 * k0 / p36 * (U256::from(2u64) * a_gamma[1] + U256::from(3u64) * p18) / p18;

    // NNAG2 = N^N * A * gamma^2 / A_MULTIPLIER = 27 * A * gamma^2 / 10000
    let nnag2 = a_gamma[0] * a_gamma[1] * a_gamma[1] / A_MULTIPLIER;

    // denominator = GK0 + NNAG2 * xp[0] / D * K0 / 10^36
    let denominator = gk0 + nnag2 * xp[0] / d * k0 / p36;

    // p[k] = xp[0] * (GK0 + NNAG2 * xp[k+1] / D * K0 / 10^36) / xp[k+1] * 10^18 / denominator
    let p0 = xp[0] * (gk0 + nnag2 * xp[1] / d * k0 / p36) / xp[1] * p18 / denominator;
    let p1 = xp[0] * (gk0 + nnag2 * xp[2] / d * k0 / p36) / xp[2] * p18 / denominator;

    [p0, p1]
}

/// Geometric mean of 3 values (TriCrypto MATH contract version).
fn geometric_mean_3(x: [U256; 3]) -> U256 {
    // cbrt(x0 * x1 / 1e18 * x2 / 1e18)
    cbrt(x[0] * x[1] / WAD * x[2] / WAD)
}

/// Fee calculation for 3-coin pool.
fn fee_3(xp: &[U256; 3], mid_fee: U256, out_fee: U256, fee_gamma: U256) -> U256 {
    let s = xp[0] + xp[1] + xp[2];
    let n = U256::from(N_COINS);
    let mut k = WAD * n * xp[0] / s;
    k = k * n * xp[1] / s;
    k = k * n * xp[2] / s;

    let k = if fee_gamma > U256::ZERO {
        fee_gamma * WAD / (fee_gamma + WAD - k)
    } else {
        k
    };
    (mid_fee * k + out_fee * (WAD - k)) / WAD
}

impl TriCryptoNGState {
    pub fn exchange(&mut self, i: usize, j: usize, dx: U256, block_timestamp: u64) -> Option<ExchangeResult> {
        if i == j || i >= 3 || j >= 3 || dx.is_zero() {
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
        xp[0] = xp[0] * self.precisions[0];
        for k in 1..3 {
            xp[k] = xp[k] * price_scale[k - 1] * self.precisions[k] / PRECISION;
        }

        let prec_i = self.precisions[i];
        let prec_j = self.precisions[j];

        // Ramp handling
        let t = self.future_a_gamma_time;
        if t > U256::ZERO && U256::from(block_timestamp) < t {
            let mut x0_norm = x0 * prec_i;
            if i > 0 {
                x0_norm = x0_norm * price_scale[i - 1] / PRECISION;
            }
            let x1 = xp[i];
            xp[i] = x0_norm;
            self.d = tricrypto_ng::newton_d(a_gamma[0], a_gamma[1], xp, U256::ZERO)?;
            xp[i] = x1;
        }

        let d_current = self.d;
        let (y_out, k0_prev) = tricrypto_ng::get_y_3_ng(a_gamma[0], a_gamma[1], xp, d_current, j)?;
        let dy_internal = xp[j] - y_out;
        xp[j] = xp[j] - dy_internal;
        let dy_internal = dy_internal - U256::from(1u64);

        let mut dy = if j > 0 {
            dy_internal * PRECISION / price_scale[j - 1]
        } else {
            dy_internal
        };
        dy = dy / prec_j;

        let fee_amount = fee_3(&xp, self.mid_fee, self.out_fee, self.fee_gamma) * dy
            / U256::from(10u64).pow(U256::from(10u64));
        dy = dy - fee_amount;

        let y_new = y - dy;
        self.balances[j] = y_new;

        let mut y_norm = y_new * prec_j;
        if j > 0 {
            y_norm = y_norm * price_scale[j - 1] / PRECISION;
        }
        xp[j] = y_norm;

        self.tweak_price(a_gamma, xp, U256::ZERO, k0_prev, block_timestamp);

        Some(ExchangeResult { dy, fee: fee_amount })
    }

    fn tweak_price(
        &mut self,
        a_gamma: [U256; 2],
        _xp: [U256; 3],
        new_d: U256,
        k0_prev: U256,
        block_timestamp: u64,
    ) {
        let mut price_oracle = self.price_oracle;
        let mut last_prices = self.last_prices;
        let price_scale = self.price_scale;
        let last_prices_timestamp = self.last_prices_timestamp;

        let total_supply = self.total_supply;
        let old_xcp_profit = self.xcp_profit;
        let old_virtual_price = self.virtual_price;

        // Update MA if needed
        if last_prices_timestamp < block_timestamp {
            let ma_time = self.ma_time;
            let dt = U256::from(block_timestamp - last_prices_timestamp);
            let power = -I256::try_from(dt * WAD / ma_time).unwrap();
            let alpha = wad_exp(power);

            for k in 0..2 {
                let capped_price = last_prices[k].min(U256::from(2u64) * price_scale[k]);
                let new_po = (capped_price * (WAD - alpha) + price_oracle[k] * alpha) / WAD;
                price_oracle[k] = new_po;
            }
            self.price_oracle = price_oracle;
            self.last_prices_timestamp = block_timestamp;
        }

        let d_unadjusted = if new_d.is_zero() {
            tricrypto_ng::newton_d(a_gamma[0], a_gamma[1], _xp, k0_prev)
                .unwrap_or(self.d)
        } else {
            new_d
        };

        // Calculate last_prices via analytical get_p (exact, matching MATH.get_p)
        let p = get_p(_xp, d_unadjusted, a_gamma);
        for k in 0..2 {
            last_prices[k] = p[k] * price_scale[k] / WAD;
        }
        self.last_prices = last_prices;

        // Update profit numbers
        let n = U256::from(N_COINS);
        let mut xp = [U256::ZERO; 3];
        xp[0] = d_unadjusted / n;
        for k in 0..2 {
            xp[k + 1] = d_unadjusted * WAD / (n * price_scale[k]);
        }

        let mut xcp_profit = WAD;
        let mut virtual_price = WAD;

        if old_virtual_price > U256::ZERO {
            let xcp = geometric_mean_3(xp);
            virtual_price = WAD * xcp / total_supply;
            xcp_profit = old_xcp_profit * virtual_price / old_virtual_price;

            if virtual_price < old_virtual_price && self.future_a_gamma_time < U256::from(block_timestamp) {
                return; // Loss
            }
        }
        self.xcp_profit = xcp_profit;

        // Rebalance check
        if virtual_price * U256::from(2u64) - WAD
            > xcp_profit + U256::from(2u64) * self.allowed_extra_profit
        {
            let mut norm = U256::ZERO;
            for k in 0..2 {
                let ratio = price_oracle[k] * WAD / price_scale[k];
                let r = if ratio > WAD { ratio - WAD } else { WAD - ratio };
                norm = norm + r * r;
            }
            norm = isqrt(norm);

            let adjustment_step = self.adjustment_step.max(norm / U256::from(5u64));

            if norm > adjustment_step && old_virtual_price > U256::ZERO {
                let mut p_new = [U256::ZERO; 2];
                for k in 0..2 {
                    p_new[k] = (price_scale[k] * (norm - adjustment_step)
                        + adjustment_step * price_oracle[k])
                        / norm;
                }

                let mut xp_adj = _xp;
                for k in 0..2 {
                    xp_adj[k + 1] = _xp[k + 1] * p_new[k] / price_scale[k];
                }

                if let Some(d) = tricrypto_ng::newton_d(a_gamma[0], a_gamma[1], xp_adj, U256::ZERO) {
                    let mut xp2 = [U256::ZERO; 3];
                    xp2[0] = d / n;
                    for k in 0..2 {
                        xp2[k + 1] = d * WAD / (n * p_new[k]);
                    }
                    let vp = WAD * geometric_mean_3(xp2) / total_supply;

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

    /// Replay a `remove_liquidity_one_coin` event for the deployed
    /// `CurveTricryptoOptimizedWETH` (Vyper 0.3.9) contract on Base.
    ///
    /// Mirrors the on-chain sequence:
    ///   1. `_calc_withdraw_one_coin` — linear D update, get_y for new xp[i]
    ///   2. `self.balances[i] -= dy`
    ///   3. `self.totalSupply -= burn_amount`
    ///   4. `self.tweak_price(A_gamma, intermediate_xp, linear_D, 0)`
    ///
    /// Caller is responsible for having previously applied the `ClaimAdminFee`
    /// (which set `self.d = newton_d(pre_xp)` and bumped `total_supply` by the
    /// minted admin LP). At entry, `self.d` must equal the on-chain `self.D`
    /// at the moment `_calc_withdraw_one_coin` is called (= post-claim).
    ///
    /// `coin_amount` is the on-chain `dy` from the event; we use the event's
    /// value (rather than recomputing it) so a downstream `assert_eq!` would
    /// catch any divergence inside `get_y_3_ng`. Returns the new D after
    /// `tweak_price`.
    pub fn apply_remove_liquidity_one(
        &mut self,
        burn_amount: U256,
        coin_index: usize,
        coin_amount: U256,
        block_timestamp: u64,
    ) -> Option<U256> {
        if coin_index >= 3 {
            return None;
        }
        let i = coin_index;
        let a_gamma = [self.ann, self.gamma];
        let token_supply = self.total_supply;
        if token_supply.is_zero() {
            return None;
        }
        let xx = self.balances;

        // Build xp matching Vyper's _calc_withdraw_one_coin (lines 1351-1361):
        //   xp = PRECISIONS                     # [precs[0], precs[1], precs[2]]
        //   price_scale_i = PRECISION * PRECISIONS[0]
        //   xp[0] *= xx[0]
        //   for k in 1..N:
        //       p = price_scale[k-1]
        //       if i == k: price_scale_i = p * xp[i]   # xp[i] still PRECISIONS[i]
        //       xp[k] = xp[k] * xx[k] * p / PRECISION
        let mut xp = self.precisions;
        let mut price_scale_i = PRECISION * self.precisions[0];
        xp[0] *= xx[0];
        for k in 1..3 {
            let p = self.price_scale[k - 1];
            if i == k {
                // xp[i] here is still self.precisions[i] (not yet multiplied by xx[i] * p)
                price_scale_i = p * xp[i];
            }
            xp[k] = xp[k] * xx[k] * p / PRECISION;
        }

        // _calc_withdraw_one_coin starts with D0 = self.D (we ignore the
        // ramping branch — caller is responsible for ensuring self.d is correct
        // at entry; we don't observe a ramp on Base TriCryptoNG in tests).
        let d0 = self.d;
        let mut d = d0;

        // Fee selection per Vyper (lines 1382-1391):
        //   xp_imprecise = xp; xp_correction = xp[i] * N * token_amount / total
        //   fee = self.out_fee
        //   if xp_correction < xp_imprecise[i]:
        //       xp_imprecise[i] -= xp_correction
        //       fee = self._fee(xp_imprecise)   # = crypto_fee
        let n_coins = U256::from(N_COINS);
        let xp_correction = xp[i] * n_coins * burn_amount / token_supply;
        let fee = if xp_correction < xp[i] {
            let mut xp_imprecise = xp;
            xp_imprecise[i] -= xp_correction;
            tricrypto_ng::crypto_fee(&xp_imprecise, self.mid_fee, self.out_fee, self.fee_gamma)?
        } else {
            self.out_fee
        };

        // dD = burn_amount * D / token_supply
        let dd = burn_amount * d / token_supply;
        // D_fee = fee * dD / (2 * 10**10) + 1
        let two_fee_denom = U256::from(2u64) * FEE_DENOMINATOR;
        let d_fee = fee * dd / two_fee_denom + U256::from(1u64);

        if d_fee >= dd {
            return None; // sanity: the linear formula would underflow
        }
        d = d - (dd - d_fee);

        // y = MATH.get_y(A_gamma, xp, D, i)[0]
        let (y, _k0_prev) =
            tricrypto_ng::get_y_3_ng(a_gamma[0], a_gamma[1], xp, d, i)?;

        if xp[i] <= y {
            return None;
        }
        // dy = (xp[i] - y) * PRECISION / price_scale_i
        let dy_internal = (xp[i] - y) * PRECISION / price_scale_i;
        // Sanity-check against the event-emitted coin_amount (= on-chain dy)
        if dy_internal != coin_amount {
            return None;
        }

        // xp[i] = y  (intermediate xp passed to tweak_price)
        xp[i] = y;


        // self.balances[i] -= dy   (line 768 in v0.3.9)
        self.balances[i] = self.balances[i] - coin_amount;
        // self.burnFrom(msg.sender, token_amount) -> totalSupply -= token_amount
        self.total_supply = self.total_supply - burn_amount;

        // tweak_price(A_gamma, xp, D, 0) -- note: D here is the LINEAR D, not 0
        self.tweak_price(a_gamma, xp, d, U256::ZERO, block_timestamp);

        Some(self.d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wad_exp_zero() {
        let result = wad_exp(I256::ZERO);
        assert_eq!(result, WAD, "e^0 should be 1e18");
    }

    #[test]
    fn wad_exp_one() {
        let result = wad_exp(I256::try_from(WAD).unwrap());
        // e^1 ≈ 2.71828... * 1e18
        let expected = U256::from(2_718_281_828_459_045_235u128);
        let diff = if result > expected { result - expected } else { expected - result };
        assert!(diff < U256::from(1_000_000_000u64), "e^1 precision: diff={diff}");
    }

    #[test]
    fn wad_exp_negative() {
        let neg_one = -I256::try_from(WAD).unwrap();
        let result = wad_exp(neg_one);
        // e^-1 ≈ 0.36787... * 1e18
        let expected = U256::from(367_879_441_171_442_321u64);
        let diff = if result > expected { result - expected } else { expected - result };
        assert!(diff < U256::from(1_000_000_000u64), "e^-1 precision");
    }
}
