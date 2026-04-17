//! TriCryptoNG — Next-gen 3-coin CryptoSwap (USDC/WBTC/WETH).
//!
//! Uses hybrid cubic+Newton solver (get_y) instead of pure Newton.
//! Vyper: https://github.com/curvefi/tricrypto-ng/blob/main/contracts/main/CurveTricryptoOptimized.vy
//!      + https://github.com/curvefi/tricrypto-ng/blob/main/contracts/main/CurveCryptoMathOptimized3.vy

use alloy_primitives::{I256, U256};

pub const WAD: U256 = U256::from_limbs([1_000_000_000_000_000_000, 0, 0, 0]);
pub const FEE_DENOMINATOR: U256 = U256::from_limbs([10_000_000_000, 0, 0, 0]);
pub const A_MULTIPLIER: U256 = U256::from_limbs([10_000, 0, 0, 0]);
const MAX_ITERATIONS: usize = 255;

pub fn isqrt(x: U256) -> U256 {
    if x.is_zero() {
        return U256::ZERO;
    }
    let mut z = (x + U256::from(1)) >> 1;
    let mut y = x;
    while z < y {
        y = z;
        z = (x / z + z) >> 1;
    }
    y
}

pub fn snekmate_log_2(x: U256) -> u32 {
    if x.is_zero() {
        return 0;
    }
    let mut value = x;
    let mut result: u32 = 0;
    if value >> 128 != U256::ZERO {
        value >>= 128;
        result = 128;
    }
    if value >> 64 != U256::ZERO {
        value >>= 64;
        result += 64;
    }
    if value >> 32 != U256::ZERO {
        value >>= 32;
        result += 32;
    }
    if value >> 16 != U256::ZERO {
        value >>= 16;
        result += 16;
    }
    if value >> 8 != U256::ZERO {
        value >>= 8;
        result += 8;
    }
    if value >> 4 != U256::ZERO {
        value >>= 4;
        result += 4;
    }
    if value >> 2 != U256::ZERO {
        value >>= 2;
        result += 2;
    }
    if value >> 1 != U256::ZERO {
        result += 1;
    }
    result
}

/// cbrt overflow threshold: 115792089237316195423570985008687907853269
const CBRT_THRESHOLD: U256 = U256::from_limbs([14562287877669245909, 5208750325433214395, 340, 0]);

pub fn cbrt(x: U256) -> U256 {
    let threshold = CBRT_THRESHOLD;
    let (xx, scale_back) = if x >= threshold * WAD {
        (x, 0u8)
    } else if x >= threshold {
        (x * WAD, 1)
    } else {
        (x * U256::from(10u128.pow(36)), 2)
    };
    let log2x = snekmate_log_2(xx);
    let remainder = (log2x % 3) as usize;
    let pow_1260: [U256; 3] = [
        U256::from(1u64),
        U256::from(1260u64),
        U256::from(1587600u64),
    ];
    let pow_1000: [U256; 3] = [
        U256::from(1u64),
        U256::from(1000u64),
        U256::from(1000000u64),
    ];
    let mut a = (U256::from(1u64) << (log2x / 3)) * pow_1260[remainder] / pow_1000[remainder];
    for _ in 0..7 {
        let a_sq = a * a;
        if a_sq.is_zero() {
            break;
        }
        a = (U256::from(2u64) * a + xx / a_sq) / U256::from(3u64);
    }
    match scale_back {
        0 => a * U256::from(1_000_000_000_000u64),
        1 => a * U256::from(1_000_000u64),
        _ => a,
    }
}

pub fn newton_y_3(ann: U256, gamma: U256, x: [U256; 3], d: U256, j: usize) -> Option<U256> {
    let n = U256::from(3u64);
    let mut others: Vec<U256> = x
        .iter()
        .enumerate()
        .filter(|(k, _)| *k != j)
        .map(|(_, v)| *v)
        .collect();
    others.sort_unstable_by(|a, b| b.cmp(a));
    let (x_0, x_1) = (others[0], others[1]);
    // Vyper: y = D/N, then for each other coin (small first): y = y*D/(x*N)
    let mut y = d / n;
    for &other in others.iter().rev() {
        y = y * d / (other * n);
    }
    let k0_i = WAD * n * x_0 / d * n * x_1 / d;
    let s_i = x_0 + x_1;
    let convergence_limit = (others.iter().max().copied().unwrap_or(U256::ZERO)
        / U256::from(10u128.pow(14)))
    .max(d / U256::from(10u128.pow(14)))
    .max(U256::from(100u64));
    let __g1k0 = gamma + WAD;
    for _ in 0..MAX_ITERATIONS {
        let y_prev = y;
        let k0 = k0_i * y * n / d;
        let s = s_i + y;
        let _g1k0 = if __g1k0 > k0 {
            __g1k0 - k0 + U256::from(1)
        } else {
            k0 - __g1k0 + U256::from(1)
        };
        let mul1 = WAD * d / gamma * _g1k0 / gamma * _g1k0 * A_MULTIPLIER / ann;
        let mul2 = WAD + U256::from(2u64) * WAD * k0 / _g1k0;
        let yfprime = WAD * y + s * mul2 + mul1;
        let _dyfprime = d * mul2;
        if yfprime < _dyfprime {
            y = y_prev / U256::from(2);
            continue;
        }
        let yfprime = yfprime - _dyfprime;
        let fprime = yfprime / y;
        let y_minus = mul1 / fprime;
        let y_plus = (yfprime + WAD * d) / fprime + y_minus * WAD / k0;
        let y_minus = y_minus + WAD * s / fprime;
        if y_plus < y_minus {
            y = y_prev / U256::from(2);
        } else {
            y = y_plus - y_minus;
        }
        let diff = if y > y_prev { y - y_prev } else { y_prev - y };
        if diff < convergence_limit.max(y / U256::from(10u128.pow(14))) {
            let frac = y * WAD / d;
            if frac < U256::from(10u128.pow(16)) || frac > U256::from(10u128.pow(20)) {
                return None;
            }
            return Some(y);
        }
    }
    None
}

pub fn get_y_3_ng(ann: U256, gamma: U256, x: [U256; 3], d: U256, i: usize) -> Option<(U256, U256)> {
    // These closures convert known small constants from the Vyper Cardano solver into I256.
    // All values are hardcoded literals (max 10^36 << 2^255), so try_from never fails.
    let s = |v: u128| -> I256 { I256::try_from(v).expect("i256 const") };
    let p = |exp: u32| -> U256 { U256::from(10u64).pow(U256::from(exp)) };
    let si = |exp: u32| -> I256 { I256::try_from(p(exp)).expect("i256 pow") };
    let (j_idx, k_idx) = match i {
        0 => (1usize, 2usize),
        1 => (0, 2),
        2 => (0, 1),
        _ => return None,
    };
    let ann_s = I256::try_from(ann).ok()?;
    let gamma_s = I256::try_from(gamma).ok()?;
    let d_s = I256::try_from(d).ok()?;
    let x_j = I256::try_from(x[j_idx]).ok()?;
    let x_k = I256::try_from(x[k_idx]).ok()?;
    let gamma2 = gamma_s.wrapping_mul(gamma_s);
    let e18 = I256::try_from(WAD).expect("WAD fits I256");
    let a_mul_s = I256::try_from(A_MULTIPLIER).expect("A_MULTIPLIER fits I256");
    let a: I256 = si(36) / s(27);
    let b: I256 = si(36) / s(9) + s(2).wrapping_mul(e18).wrapping_mul(gamma_s) / s(27)
        - d_s.wrapping_mul(d_s) / x_j * gamma2 * ann_s / s(27 * 27) / a_mul_s / x_k;
    let c: I256 = si(36) / s(9)
        + gamma_s.wrapping_mul(gamma_s + s(4).wrapping_mul(e18)) / s(27)
        + gamma2 * (x_j + x_k - d_s) / d_s * ann_s / s(27) / a_mul_s;
    let d_coeff: I256 = (e18 + gamma_s).wrapping_mul(e18 + gamma_s) / s(27);
    let d0: I256 = (s(3).wrapping_mul(a).wrapping_mul(c) / b - b).abs();
    let d0_u = U256::try_from(d0).unwrap_or(U256::ZERO);
    let divider: I256 = if d0_u > p(48) {
        si(30)
    } else if d0_u > p(44) {
        si(26)
    } else if d0_u > p(40) {
        si(22)
    } else if d0_u > p(36) {
        si(18)
    } else if d0_u > p(32) {
        si(14)
    } else if d0_u > p(28) {
        si(10)
    } else if d0_u > p(24) {
        si(6)
    } else if d0_u > p(20) {
        si(2)
    } else {
        s(1)
    };
    let (a, b, c, d_coeff) = if a.abs() > b.abs() {
        let ap = (a / b).abs();
        (
            a.wrapping_mul(ap) / divider,
            (b * ap) / divider,
            (c * ap) / divider,
            (d_coeff * ap) / divider,
        )
    } else {
        let ap = (b / a).abs();
        (
            a / ap / divider,
            b / ap / divider,
            c / ap / divider,
            d_coeff / ap / divider,
        )
    };
    let _3ac = s(3).wrapping_mul(a).wrapping_mul(c);
    let delta0 = _3ac / b - b;
    let delta1 = s(3).wrapping_mul(_3ac) / b
        - s(2).wrapping_mul(b)
        - s(27).wrapping_mul(a.wrapping_mul(a)) / b * d_coeff / b;
    let sqrt_arg =
        delta1.wrapping_mul(delta1) + s(4).wrapping_mul(delta0.wrapping_mul(delta0)) / b * delta0;
    if sqrt_arg <= I256::ZERO {
        let y = newton_y_3(ann, gamma, x, d, i)?;
        return Some((y, U256::ZERO));
    }
    let sqrt_val = I256::try_from(isqrt(U256::try_from(sqrt_arg).ok()?)).ok()?;
    let b_cbrt: I256 = if b >= I256::ZERO {
        I256::try_from(cbrt(U256::try_from(b).ok()?)).ok()?
    } else {
        -I256::try_from(cbrt(U256::try_from(-b).ok()?)).ok()?
    };
    let second_cbrt: I256 = if delta1 > I256::ZERO {
        I256::try_from(cbrt(
            U256::try_from(delta1 + sqrt_val).ok()? / U256::from(2u64),
        ))
        .ok()?
    } else {
        -I256::try_from(cbrt(
            U256::try_from(-(delta1 - sqrt_val)).ok()? / U256::from(2u64),
        ))
        .ok()?
    };
    let c1: I256 = b_cbrt
        .wrapping_mul(b_cbrt)
        .wrapping_div(e18)
        .wrapping_mul(second_cbrt)
        .wrapping_div(e18);
    let root_k0: I256 = (b + b * delta0 / c1 - c1) / s(3);
    let root: I256 = d_s.wrapping_mul(d_s) / s(27) / x_k * d_s / x_j * root_k0 / a;
    let y_out = U256::try_from(root).ok()?;
    let k0_prev = U256::try_from(e18.wrapping_mul(root_k0) / a).ok()?;
    let frac = y_out * WAD / d;
    if frac < p(16) - U256::from(1) || frac >= p(20) + U256::from(1) {
        return None;
    }
    Some((y_out, k0_prev))
}

pub fn crypto_fee(xp: &[U256], mid_fee: U256, out_fee: U256, fee_gamma: U256) -> Option<U256> {
    let s: U256 = xp
        .iter()
        .try_fold(U256::ZERO, |acc, v| acc.checked_add(*v))?;
    if s.is_zero() {
        return None;
    }
    let n = U256::from(xp.len());
    let mut k = WAD;
    for x_i in xp {
        k = k * n * (*x_i) / s;
    }
    let f = if fee_gamma > U256::ZERO {
        fee_gamma * WAD / (fee_gamma + WAD - k)
    } else {
        k
    };
    Some((mid_fee * f + out_fee * (WAD - f)) / WAD)
}

/// Sort 3 values descending.
fn sort3_desc(x: [U256; 3]) -> [U256; 3] {
    let mut s = x;
    if s[0] < s[1] { s.swap(0, 1); }
    if s[1] < s[2] { s.swap(1, 2); }
    if s[0] < s[1] { s.swap(0, 1); }
    s
}

/// Geometric mean of 3 values, WAD-scaled: cbrt(x[0] * x[1] * x[2]).
fn geometric_mean_3(x: [U256; 3]) -> U256 {
    // _geometric_mean in Vyper: cbrt of product, avoiding overflow via
    // intermediate division. We match the on-chain implementation:
    // cbrt(x[0] * x[1] / 10^18 * x[2] / 10^18) * 10^12
    // but the actual Vyper code uses a different approach:
    // D = isqrt(x[0] * x[1] / 10^18) * isqrt(x[2] * 10^18)
    // Actually the on-chain code uses:
    // _geometric_mean(x) which calls cbrt(x0 * x1 / 1e18 * x2 / 1e18) * 1e12
    // Let's use the exact on-chain formula from TricryptoMath.vy:
    // return self._cbrt(unsafe_div(unsafe_div(x_sorted[0] * x_sorted[1], 10**18) * x_sorted[2], 10**18))
    cbrt(x[0] * x[1] / WAD * x[2] / WAD)
}

/// Compute the CryptoSwap invariant D using Newton's method (3-coin).
///
/// Port of `CurveCryptoMathOptimized3::newton_D`. ANN is `A * N^N`
/// (already scaled by `A_MULTIPLIER`). `x` are the normalized balances.
/// `k0_prev` is an initial guess hint; pass `U256::ZERO` for default.
///
/// Returns `None` if the iteration does not converge.
pub fn newton_d(ann: U256, gamma: U256, x_unsorted: [U256; 3], k0_prev: U256) -> Option<U256> {
    let n = U256::from(3u64);
    let x = sort3_desc(x_unsorted);

    // Safety: x[0] must be in valid range
    // x[0] < max_value / 10^18 * N^N
    if x[0].is_zero() {
        return None;
    }

    let s = x[0] + x[1] + x[2];

    // Initial D guess
    let mut d = if k0_prev.is_zero() {
        n * geometric_mean_3(x)
    } else {
        // D = cbrt(x[0] * x[1] / K0_prev * x[2] * 27)
        // Adjusted for scale to avoid overflow
        let p18 = U256::from(10u64).pow(U256::from(18u64));
        let p24 = U256::from(10u64).pow(U256::from(24u64));
        let p36 = U256::from(10u64).pow(U256::from(36u64));
        if s > p36 {
            cbrt(x[0] * x[1] / p36 * x[2] / k0_prev * U256::from(27u64) * U256::from(10u64).pow(U256::from(12u64)))
        } else if s > p24 {
            cbrt(x[0] * x[1] / p24 * x[2] / k0_prev * U256::from(27u64) * U256::from(10u64).pow(U256::from(6u64)))
        } else {
            cbrt(x[0] * x[1] / p18 * x[2] / k0_prev * U256::from(27u64))
        }
    };

    let g1k0_base = gamma + WAD;

    for _ in 0..MAX_ITERATIONS {
        let d_prev = d;
        if d.is_zero() {
            return None;
        }

        // K0 = 10^18 * x[0] * N / D * x[1] * N / D * x[2] * N / D
        let k0 = WAD * x[0] * n / d * x[1] * n / d * x[2] * n / d;

        let _g1k0 = if g1k0_base > k0 {
            g1k0_base - k0 + U256::from(1u64)
        } else {
            k0 - g1k0_base + U256::from(1u64)
        };

        // mul1 = 10^18 * D / gamma * _g1k0 / gamma * _g1k0 * A_MULTIPLIER / ANN
        let mul1 = WAD * d / gamma * _g1k0 / gamma * _g1k0 * A_MULTIPLIER / ann;

        // mul2 = (2 * 10^18) * N * K0 / _g1k0
        let mul2 = U256::from(2u64) * WAD * n * k0 / _g1k0;

        // neg_fprime = (S + S * mul2 / 10^18) + mul1 * N / K0 - mul2 * D / 10^18
        if k0.is_zero() {
            return None;
        }
        let neg_fprime = (s + s * mul2 / WAD) + mul1 * n / k0 - mul2 * d / WAD;

        if neg_fprime.is_zero() {
            return None;
        }

        let d_plus = d * (neg_fprime + s) / neg_fprime;
        let mut d_minus = d * d / neg_fprime;

        if WAD > k0 {
            d_minus += d * (mul1 / neg_fprime) / WAD * (WAD - k0) / k0;
        } else {
            d_minus -= d * (mul1 / neg_fprime) / WAD * (k0 - WAD) / k0;
        }

        d = if d_plus > d_minus {
            d_plus - d_minus
        } else {
            (d_minus - d_plus) / U256::from(2u64)
        };

        let diff = if d > d_prev { d - d_prev } else { d_prev - d };

        let threshold = U256::from(10u64).pow(U256::from(16u64)).max(d);
        if diff * U256::from(10u64).pow(U256::from(14u64)) < threshold {
            // Validate output fractions
            for &xi in &x {
                let frac = xi * WAD / d;
                let min_frac = U256::from(10u64).pow(U256::from(16u64)) - U256::from(1u64);
                let max_frac = U256::from(10u64).pow(U256::from(20u64)) + U256::from(1u64);
                if frac < min_frac || frac >= max_frac {
                    return None;
                }
            }
            return Some(d);
        }
    }
    None // Did not converge
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isqrt_known_values() {
        assert_eq!(isqrt(U256::from(16u64)), U256::from(4u64));
        assert_eq!(isqrt(U256::from(25u64)), U256::from(5u64));
        assert_eq!(isqrt(U256::from(26u64)), U256::from(5u64));
    }

    #[test]
    fn cbrt_monotonic() {
        let a = U256::from(1_000_000_000_000_000_000u128);
        let b = U256::from(27_000_000_000_000_000_000u128);
        let ca = cbrt(a);
        let cb = cbrt(b);
        assert!(cb > ca);
    }

    fn realistic_params() -> (U256, U256, [U256; 3], U256) {
        let wad = WAD;
        let ann = U256::from(1707629u64) * A_MULTIPLIER;
        let gamma = U256::from(11_809_167_828_997u64);
        let balance = U256::from(10_000u64) * wad;
        let x = [balance, balance, balance];
        let d = U256::from(30_000u64) * wad;
        (ann, gamma, x, d)
    }

    #[test]
    fn newton_d_3_balanced() {
        let wad = WAD;
        let ann = U256::from(1707629u64) * A_MULTIPLIER;
        let gamma = U256::from(11_809_167_828_997u64);
        let balance = U256::from(10_000u64) * wad;
        let d = newton_d(ann, gamma, [balance, balance, balance], U256::ZERO).expect("converge");
        assert!(d > U256::ZERO);
        // For balanced 3-coin pool, D ≈ 3 * balance
        let s = balance * U256::from(3u64);
        assert!(d <= s + wad);
    }

    #[test]
    fn newton_d_3_idempotent() {
        let wad = WAD;
        let ann = U256::from(1707629u64) * A_MULTIPLIER;
        let gamma = U256::from(11_809_167_828_997u64);
        let x = [U256::from(10_000u64) * wad, U256::from(10_000u64) * wad, U256::from(10_000u64) * wad];
        let d1 = newton_d(ann, gamma, x, U256::ZERO).expect("converge");
        let d2 = newton_d(ann, gamma, x, U256::ZERO).expect("converge");
        assert_eq!(d1, d2);
    }

    #[test]
    fn newton_d_3_monotonic() {
        let wad = WAD;
        let ann = U256::from(1707629u64) * A_MULTIPLIER;
        let gamma = U256::from(11_809_167_828_997u64);
        let d1 = newton_d(ann, gamma, [U256::from(10_000u64) * wad, U256::from(10_000u64) * wad, U256::from(10_000u64) * wad], U256::ZERO).unwrap();
        let d2 = newton_d(ann, gamma, [U256::from(12_000u64) * wad, U256::from(10_000u64) * wad, U256::from(10_000u64) * wad], U256::ZERO).unwrap();
        assert!(d2 > d1);
    }

    #[test]
    fn newton_d_3_matches_onchain_base_pool() {
        // Differential test against on-chain MATH.newton_D for a real TriCryptoNG pool on Base.
        // Pool: 0xd48949347efe6029A9F7bb8F97E78C423F88486E
        // MATH: 0x5373E1B9f2781099f6796DFe5D68DE59ac2F18E3
        // A=2700000, gamma=1300000000000, precisions=[1,1,1]
        // xp = [1564325938278762767729778, 1674558251010830450444325, 1814103230935323407441172]
        // On-chain MATH.newton_D(2700000, 1300000000000, xp, 0) = 5043877571863252725139907
        let ann = U256::from(2_700_000u64);
        let gamma = U256::from(1_300_000_000_000u64);
        let xp = [
            U256::from_str_radix("1564325938278762767729778", 10).unwrap(),
            U256::from_str_radix("1674558251010830450444325", 10).unwrap(),
            U256::from_str_radix("1814103230935323407441172", 10).unwrap(),
        ];
        let expected_d = U256::from_str_radix("5043877571863252725139907", 10).unwrap();

        let d = newton_d(ann, gamma, xp, U256::ZERO).expect("should converge");
        assert_eq!(d, expected_d, "must match on-chain MATH.newton_D exactly");
    }

    #[test]
    fn newton_y_3_convergence() {
        let (ann, gamma, x, d) = realistic_params();
        let y = newton_y_3(ann, gamma, x, d, 0).expect("converge");
        assert!(y > U256::ZERO);
        assert!(y < d);
    }

    #[test]
    fn get_y_3_ng_convergence() {
        let (ann, gamma, x, d) = realistic_params();
        let result = get_y_3_ng(ann, gamma, x, d, 2);
        assert!(result.is_some());
        let (y, _k0) = result.expect("converge");
        assert!(y > U256::ZERO);
        assert!(y < d);
    }

    #[test]
    fn crypto_fee_three_coins_balanced() {
        let wad = WAD;
        let mid_fee = U256::from(3_000_000u64);
        let out_fee = U256::from(30_000_000u64);
        let fee_gamma = U256::from(230_000_000_000_000u64);
        let xp = [
            U256::from(100_000u64) * wad,
            U256::from(100_000u64) * wad,
            U256::from(100_000u64) * wad,
        ];
        let fee = crypto_fee(&xp, mid_fee, out_fee, fee_gamma).expect("fee");
        assert!(fee >= mid_fee);
        assert!(fee < out_fee);
    }

    #[test]
    fn get_y_3_ng_swap_reduces() {
        let wad = WAD;
        let (ann, gamma, x, d) = realistic_params();
        let dx = U256::from(10u64) * wad;
        let (y_before, _) = get_y_3_ng(ann, gamma, x, d, 2).expect("before");
        let (y_after, _) = get_y_3_ng(ann, gamma, [x[0] + dx, x[1], x[2]], d, 2).expect("after");
        assert!(y_after < y_before);
    }
}
