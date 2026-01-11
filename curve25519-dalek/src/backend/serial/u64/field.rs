// -*- mode: rust; -*-
//
// This file is part of curve25519-dalek.
// Copyright (c) 2016-2021 isis lovecruft
// Copyright (c) 2016-2019 Henry de Valence
// See LICENSE for licensing information.
//
// Authors:
// - isis agora lovecruft <isis@patternsinthevoid.net>
// - Henry de Valence <hdevalence@hdevalence.ca>

//! Field arithmetic modulo \\(p = 2\^{255} - 19\\), using \\(64\\)-bit
//! limbs with \\(128\\)-bit products.

use core::fmt::Debug;
use core::ops::Neg;
use core::ops::{Add, AddAssign};
use core::ops::{Mul, MulAssign};
use core::ops::{Sub, SubAssign};

use subtle::Choice;
use subtle::ConditionallySelectable;

use multiversion::multiversion;

#[cfg(feature = "zeroize")]
use zeroize::Zeroize;

/// A `FieldElement51` represents an element of the field
/// \\( \mathbb Z / (2\^{255} - 19)\\).
///
/// In the 64-bit implementation, a `FieldElement` is represented in
/// radix \\(2\^{51}\\) as five `u64`s; the coefficients are allowed to
/// grow up to \\(2\^{54}\\) between reductions modulo \\(p\\).
///
/// # Note
///
/// The `curve25519_dalek::field` module provides a type alias
/// `curve25519_dalek::field::FieldElement` to either `FieldElement51`
/// or `FieldElement2625`.
///
/// The backend-specific type `FieldElement51` should not be used
/// outside of the `curve25519_dalek::field` module.
#[derive(Copy, Clone)]
pub struct FieldElement51(pub(crate) [u64; 5]);

impl Debug for FieldElement51 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "FieldElement51({:?})", &self.0[..])
    }
}

#[cfg(feature = "zeroize")]
impl Zeroize for FieldElement51 {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

impl<'a> AddAssign<&'a FieldElement51> for FieldElement51 {
    fn add_assign(&mut self, _rhs: &'a FieldElement51) {
        for i in 0..5 {
            self.0[i] += _rhs.0[i];
        }
    }
}

impl<'a> Add<&'a FieldElement51> for &FieldElement51 {
    type Output = FieldElement51;
    fn add(self, _rhs: &'a FieldElement51) -> FieldElement51 {
        let mut output = *self;
        output += _rhs;
        output
    }
}

impl<'a> SubAssign<&'a FieldElement51> for FieldElement51 {
    fn sub_assign(&mut self, _rhs: &'a FieldElement51) {
        let result = (self as &FieldElement51) - _rhs;
        self.0 = result.0;
    }
}

impl<'a> Sub<&'a FieldElement51> for &FieldElement51 {
    type Output = FieldElement51;
    fn sub(self, _rhs: &'a FieldElement51) -> FieldElement51 {
        // To avoid underflow, first add a multiple of p.
        // Choose 16*p = p << 4 to be larger than 54-bit _rhs.
        //
        // If we could statically track the bitlengths of the limbs
        // of every FieldElement51, we could choose a multiple of p
        // just bigger than _rhs and avoid having to do a reduction.
        //
        // Since we don't yet have type-level integers to do this, we
        // have to add an explicit reduction call here.
        FieldElement51::reduce([
            (self.0[0] + 36028797018963664u64) - _rhs.0[0],
            (self.0[1] + 36028797018963952u64) - _rhs.0[1],
            (self.0[2] + 36028797018963952u64) - _rhs.0[2],
            (self.0[3] + 36028797018963952u64) - _rhs.0[3],
            (self.0[4] + 36028797018963952u64) - _rhs.0[4],
        ])
    }
}

impl<'a> MulAssign<&'a FieldElement51> for FieldElement51 {
    fn mul_assign(&mut self, _rhs: &'a FieldElement51) {
        let result = (self as &FieldElement51) * _rhs;
        self.0 = result.0;
    }
}

impl<'a> Mul<&'a FieldElement51> for &FieldElement51 {
    type Output = FieldElement51;

    #[rustfmt::skip] // keep alignment of c* calculations
    fn mul(self, _rhs: &'a FieldElement51) -> FieldElement51 {
        /// Helper function to multiply two 64-bit integers with 128
        /// bits of output.
        #[inline(always)]
        fn m(x: u64, y: u64) -> u128 { (x as u128) * (y as u128) }

        // Alias self, _rhs for more readable formulas
        let a: &[u64; 5] = &self.0;
        let b: &[u64; 5] = &_rhs.0;

        // Precondition: assume input limbs a[i], b[i] are bounded as
        //
        // a[i], b[i] < 2^(51 + b)
        //
        // where b is a real parameter measuring the "bit excess" of the limbs.

        // 64-bit precomputations to avoid 128-bit multiplications.
        //
        // This fits into a u64 whenever 51 + b + lg(19) < 64.
        //
        // Since 51 + b + lg(19) < 51 + 4.25 + b
        //                       = 55.25 + b,
        // this fits if b < 8.75.
        let b1_19 = b[1] * 19;
        let b2_19 = b[2] * 19;
        let b3_19 = b[3] * 19;
        let b4_19 = b[4] * 19;

        // Multiply to get 128-bit coefficients of output
        let     c0: u128 = m(a[0], b[0]) + m(a[4], b1_19) + m(a[3], b2_19) + m(a[2], b3_19) + m(a[1], b4_19);
        let mut c1: u128 = m(a[1], b[0]) + m(a[0],  b[1]) + m(a[4], b2_19) + m(a[3], b3_19) + m(a[2], b4_19);
        let mut c2: u128 = m(a[2], b[0]) + m(a[1],  b[1]) + m(a[0],  b[2]) + m(a[4], b3_19) + m(a[3], b4_19);
        let mut c3: u128 = m(a[3], b[0]) + m(a[2],  b[1]) + m(a[1],  b[2]) + m(a[0],  b[3]) + m(a[4], b4_19);
        let mut c4: u128 = m(a[4], b[0]) + m(a[3],  b[1]) + m(a[2],  b[2]) + m(a[1],  b[3]) + m(a[0] , b[4]);

        // How big are the c[i]? We have
        //
        //    c[i] < 2^(102 + 2*b) * (1+i + (4-i)*19)
        //         < 2^(102 + lg(1 + 4*19) + 2*b)
        //         < 2^(108.27 + 2*b)
        //
        // The carry (c[i] >> 51) fits into a u64 when
        //    108.27 + 2*b - 51 < 64
        //    2*b < 6.73
        //    b < 3.365.
        //
        // So we require b < 3 to ensure this fits.
        debug_assert!(a[0] < (1 << 54)); debug_assert!(b[0] < (1 << 54));
        debug_assert!(a[1] < (1 << 54)); debug_assert!(b[1] < (1 << 54));
        debug_assert!(a[2] < (1 << 54)); debug_assert!(b[2] < (1 << 54));
        debug_assert!(a[3] < (1 << 54)); debug_assert!(b[3] < (1 << 54));
        debug_assert!(a[4] < (1 << 54)); debug_assert!(b[4] < (1 << 54));

        // Casting to u64 and back tells the compiler that the carry is
        // bounded by 2^64, so that the addition is a u128 + u64 rather
        // than u128 + u128.

        const LOW_51_BIT_MASK: u64 = (1u64 << 51) - 1;
        let mut out = [0u64; 5];

        c1 += ((c0 >> 51) as u64) as u128;
        out[0] = (c0 as u64) & LOW_51_BIT_MASK;

        c2 += ((c1 >> 51) as u64) as u128;
        out[1] = (c1 as u64) & LOW_51_BIT_MASK;

        c3 += ((c2 >> 51) as u64) as u128;
        out[2] = (c2 as u64) & LOW_51_BIT_MASK;

        c4 += ((c3 >> 51) as u64) as u128;
        out[3] = (c3 as u64) & LOW_51_BIT_MASK;

        let carry: u64 = (c4 >> 51) as u64;
        out[4] = (c4 as u64) & LOW_51_BIT_MASK;

        // To see that this does not overflow, we need out[0] + carry * 19 < 2^64.
        //
        // c4 < a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0 + (carry from c3)
        //    < 5*(2^(51 + b) * 2^(51 + b)) + (carry from c3)
        //    < 2^(102 + 2*b + lg(5)) + 2^64.
        //
        // When b < 3 we get
        //
        // c4 < 2^110.33  so that carry < 2^59.33
        //
        // so that
        //
        // out[0] + carry * 19 < 2^51 + 19 * 2^59.33 < 2^63.58
        //
        // and there is no overflow.
        out[0] += carry * 19;

        // Now out[1] < 2^51 + 2^(64 -51) = 2^51 + 2^13 < 2^(51 + epsilon).
        out[1] += out[0] >> 51;
        out[0] &= LOW_51_BIT_MASK;

        // Now out[i] < 2^(51 + epsilon) for all i.
        FieldElement51(out)
    }
}

impl Neg for &FieldElement51 {
    type Output = FieldElement51;
    fn neg(self) -> FieldElement51 {
        let mut output = *self;
        output.negate();
        output
    }
}

impl ConditionallySelectable for FieldElement51 {
    fn conditional_select(
        a: &FieldElement51,
        b: &FieldElement51,
        choice: Choice,
    ) -> FieldElement51 {
        FieldElement51([
            u64::conditional_select(&a.0[0], &b.0[0], choice),
            u64::conditional_select(&a.0[1], &b.0[1], choice),
            u64::conditional_select(&a.0[2], &b.0[2], choice),
            u64::conditional_select(&a.0[3], &b.0[3], choice),
            u64::conditional_select(&a.0[4], &b.0[4], choice),
        ])
    }

    fn conditional_swap(a: &mut FieldElement51, b: &mut FieldElement51, choice: Choice) {
        u64::conditional_swap(&mut a.0[0], &mut b.0[0], choice);
        u64::conditional_swap(&mut a.0[1], &mut b.0[1], choice);
        u64::conditional_swap(&mut a.0[2], &mut b.0[2], choice);
        u64::conditional_swap(&mut a.0[3], &mut b.0[3], choice);
        u64::conditional_swap(&mut a.0[4], &mut b.0[4], choice);
    }

    fn conditional_assign(&mut self, other: &FieldElement51, choice: Choice) {
        self.0[0].conditional_assign(&other.0[0], choice);
        self.0[1].conditional_assign(&other.0[1], choice);
        self.0[2].conditional_assign(&other.0[2], choice);
        self.0[3].conditional_assign(&other.0[3], choice);
        self.0[4].conditional_assign(&other.0[4], choice);
    }
}

impl FieldElement51 {
    pub(crate) const fn from_limbs(limbs: [u64; 5]) -> FieldElement51 {
        FieldElement51(limbs)
    }

    /// The scalar \\( 0 \\).
    pub const ZERO: FieldElement51 = FieldElement51::from_limbs([0, 0, 0, 0, 0]);
    /// The scalar \\( 1 \\).
    pub const ONE: FieldElement51 = FieldElement51::from_limbs([1, 0, 0, 0, 0]);
    /// The scalar \\( -1 \\).
    pub const MINUS_ONE: FieldElement51 = FieldElement51::from_limbs([
        2251799813685228,
        2251799813685247,
        2251799813685247,
        2251799813685247,
        2251799813685247,
    ]);

    /// Invert the sign of this field element
    pub fn negate(&mut self) {
        // See commentary in the Sub impl
        let neg = FieldElement51::reduce([
            36028797018963664u64 - self.0[0],
            36028797018963952u64 - self.0[1],
            36028797018963952u64 - self.0[2],
            36028797018963952u64 - self.0[3],
            36028797018963952u64 - self.0[4],
        ]);
        self.0 = neg.0;
    }

    /// Given 64-bit input limbs, reduce to enforce the bound 2^(51 + epsilon).
    #[inline(always)]
    fn reduce(mut limbs: [u64; 5]) -> FieldElement51 {
        const LOW_51_BIT_MASK: u64 = (1u64 << 51) - 1;

        // Since the input limbs are bounded by 2^64, the biggest
        // carry-out is bounded by 2^13.
        //
        // The biggest carry-in is c4 * 19, resulting in
        //
        // 2^51 + 19*2^13 < 2^51.0000000001
        //
        // Because we don't need to canonicalize, only to reduce the
        // limb sizes, it's OK to do a "weak reduction", where we
        // compute the carry-outs in parallel.

        let c0 = limbs[0] >> 51;
        let c1 = limbs[1] >> 51;
        let c2 = limbs[2] >> 51;
        let c3 = limbs[3] >> 51;
        let c4 = limbs[4] >> 51;

        limbs[0] &= LOW_51_BIT_MASK;
        limbs[1] &= LOW_51_BIT_MASK;
        limbs[2] &= LOW_51_BIT_MASK;
        limbs[3] &= LOW_51_BIT_MASK;
        limbs[4] &= LOW_51_BIT_MASK;

        limbs[0] += c4 * 19;
        limbs[1] += c0;
        limbs[2] += c1;
        limbs[3] += c2;
        limbs[4] += c3;

        FieldElement51(limbs)
    }

    /// Load a `FieldElement51` from the low 255 bits of a 256-bit
    /// input.
    ///
    /// # Warning
    ///
    /// This function does not check that the input used the canonical
    /// representative.  It masks the high bit, but it will happily
    /// decode 2^255 - 18 to 1.  Applications that require a canonical
    /// encoding of every field element should decode, re-encode to
    /// the canonical encoding, and check that the input was
    /// canonical.
    ///
    #[rustfmt::skip] // keep alignment of bit shifts
    pub const fn from_bytes(bytes: &[u8; 32]) -> FieldElement51 {
        const fn load8_at(input: &[u8], i: usize) -> u64 {
               (input[i] as u64)
            | ((input[i + 1] as u64) << 8)
            | ((input[i + 2] as u64) << 16)
            | ((input[i + 3] as u64) << 24)
            | ((input[i + 4] as u64) << 32)
            | ((input[i + 5] as u64) << 40)
            | ((input[i + 6] as u64) << 48)
            | ((input[i + 7] as u64) << 56)
        }

        let low_51_bit_mask = (1u64 << 51) - 1;
        FieldElement51(
        // load bits [  0, 64), no shift
        [  load8_at(bytes,  0)        & low_51_bit_mask
        // load bits [ 48,112), shift to [ 51,112)
        , (load8_at(bytes,  6) >>  3) & low_51_bit_mask
        // load bits [ 96,160), shift to [102,160)
        , (load8_at(bytes, 12) >>  6) & low_51_bit_mask
        // load bits [152,216), shift to [153,216)
        , (load8_at(bytes, 19) >>  1) & low_51_bit_mask
        // load bits [192,256), shift to [204,112)
        , (load8_at(bytes, 24) >> 12) & low_51_bit_mask
        ])
    }

    /// Serialize this `FieldElement51` to a 32-byte array.  The
    /// encoding is canonical.
    #[rustfmt::skip] // keep alignment of s[*] calculations
    pub fn to_bytes(self) -> [u8; 32] {
        // Let h = limbs[0] + limbs[1]*2^51 + ... + limbs[4]*2^204.
        //
        // Write h = pq + r with 0 <= r < p.
        //
        // We want to compute r = h mod p.
        //
        // If h < 2*p = 2^256 - 38,
        // then q = 0 or 1,
        //
        // with q = 0 when h < p
        //  and q = 1 when h >= p.
        //
        // Notice that h >= p <==> h + 19 >= p + 19 <==> h + 19 >= 2^255.
        // Therefore q can be computed as the carry bit of h + 19.

        // First, reduce the limbs to ensure h < 2*p.
        let mut limbs = FieldElement51::reduce(self.0).0;

        let mut q = (limbs[0] + 19) >> 51;
        q = (limbs[1] + q) >> 51;
        q = (limbs[2] + q) >> 51;
        q = (limbs[3] + q) >> 51;
        q = (limbs[4] + q) >> 51;

        // Now we can compute r as r = h - pq = r - (2^255-19)q = r + 19q - 2^255q

        limbs[0] += 19 * q;

        // Now carry the result to compute r + 19q ...
        let low_51_bit_mask = (1u64 << 51) - 1;
        limbs[1] += limbs[0] >> 51;
        limbs[0] &= low_51_bit_mask;
        limbs[2] += limbs[1] >> 51;
        limbs[1] &= low_51_bit_mask;
        limbs[3] += limbs[2] >> 51;
        limbs[2] &= low_51_bit_mask;
        limbs[4] += limbs[3] >> 51;
        limbs[3] &= low_51_bit_mask;
        // ... but instead of carrying (limbs[4] >> 51) = 2^255q
        // into another limb, discard it, subtracting the value
        limbs[4] &= low_51_bit_mask;

        // Now arrange the bits of the limbs.
        let mut s = [0u8;32];
        s[ 0] =   limbs[0]                           as u8;
        s[ 1] =  (limbs[0] >>  8)                    as u8;
        s[ 2] =  (limbs[0] >> 16)                    as u8;
        s[ 3] =  (limbs[0] >> 24)                    as u8;
        s[ 4] =  (limbs[0] >> 32)                    as u8;
        s[ 5] =  (limbs[0] >> 40)                    as u8;
        s[ 6] = ((limbs[0] >> 48) | (limbs[1] << 3)) as u8;
        s[ 7] =  (limbs[1] >>  5)                    as u8;
        s[ 8] =  (limbs[1] >> 13)                    as u8;
        s[ 9] =  (limbs[1] >> 21)                    as u8;
        s[10] =  (limbs[1] >> 29)                    as u8;
        s[11] =  (limbs[1] >> 37)                    as u8;
        s[12] = ((limbs[1] >> 45) | (limbs[2] << 6)) as u8;
        s[13] =  (limbs[2] >>  2)                    as u8;
        s[14] =  (limbs[2] >> 10)                    as u8;
        s[15] =  (limbs[2] >> 18)                    as u8;
        s[16] =  (limbs[2] >> 26)                    as u8;
        s[17] =  (limbs[2] >> 34)                    as u8;
        s[18] =  (limbs[2] >> 42)                    as u8;
        s[19] = ((limbs[2] >> 50) | (limbs[3] << 1)) as u8;
        s[20] =  (limbs[3] >>  7)                    as u8;
        s[21] =  (limbs[3] >> 15)                    as u8;
        s[22] =  (limbs[3] >> 23)                    as u8;
        s[23] =  (limbs[3] >> 31)                    as u8;
        s[24] =  (limbs[3] >> 39)                    as u8;
        s[25] = ((limbs[3] >> 47) | (limbs[4] << 4)) as u8;
        s[26] =  (limbs[4] >>  4)                    as u8;
        s[27] =  (limbs[4] >> 12)                    as u8;
        s[28] =  (limbs[4] >> 20)                    as u8;
        s[29] =  (limbs[4] >> 28)                    as u8;
        s[30] =  (limbs[4] >> 36)                    as u8;
        s[31] =  (limbs[4] >> 44)                    as u8;

        // High bit should be zero.
        debug_assert!((s[31] & 0b1000_0000u8) == 0u8);

        s
    }

    /// Given `k > 0`, return `self^(2^k)`.
    #[rustfmt::skip] // keep alignment of c* calculations
    pub fn pow2k(&self, mut k: u32) -> FieldElement51 {

        debug_assert!( k > 0 );

        /// Multiply two 64-bit integers with 128 bits of output.
        #[inline(always)]
        fn m(x: u64, y: u64) -> u128 {
            (x as u128) * (y as u128)
        }

        let mut a: [u64; 5] = self.0;

        loop {
            // Precondition: assume input limbs a[i] are bounded as
            //
            // a[i] < 2^(51 + b)
            //
            // where b is a real parameter measuring the "bit excess" of the limbs.

            // Precomputation: 64-bit multiply by 19.
            //
            // This fits into a u64 whenever 51 + b + lg(19) < 64.
            //
            // Since 51 + b + lg(19) < 51 + 4.25 + b
            //                       = 55.25 + b,
            // this fits if b < 8.75.
            let a3_19 = 19 * a[3];
            let a4_19 = 19 * a[4];

            // Multiply to get 128-bit coefficients of output.
            //
            // The 128-bit multiplications by 2 turn into 1 slr + 1 slrd each,
            // which doesn't seem any better or worse than doing them as precomputations
            // on the 64-bit inputs.
            let     c0: u128 = m(a[0],  a[0]) + 2*( m(a[1], a4_19) + m(a[2], a3_19) );
            let mut c1: u128 = m(a[3], a3_19) + 2*( m(a[0],  a[1]) + m(a[2], a4_19) );
            let mut c2: u128 = m(a[1],  a[1]) + 2*( m(a[0],  a[2]) + m(a[4], a3_19) );
            let mut c3: u128 = m(a[4], a4_19) + 2*( m(a[0],  a[3]) + m(a[1],  a[2]) );
            let mut c4: u128 = m(a[2],  a[2]) + 2*( m(a[0],  a[4]) + m(a[1],  a[3]) );

            // Same bound as in multiply:
            //    c[i] < 2^(102 + 2*b) * (1+i + (4-i)*19)
            //         < 2^(102 + lg(1 + 4*19) + 2*b)
            //         < 2^(108.27 + 2*b)
            //
            // The carry (c[i] >> 51) fits into a u64 when
            //    108.27 + 2*b - 51 < 64
            //    2*b < 6.73
            //    b < 3.365.
            //
            // So we require b < 3 to ensure this fits.
            debug_assert!(a[0] < (1 << 54));
            debug_assert!(a[1] < (1 << 54));
            debug_assert!(a[2] < (1 << 54));
            debug_assert!(a[3] < (1 << 54));
            debug_assert!(a[4] < (1 << 54));

            const LOW_51_BIT_MASK: u64 = (1u64 << 51) - 1;

            // Casting to u64 and back tells the compiler that the carry is bounded by 2^64, so
            // that the addition is a u128 + u64 rather than u128 + u128.
            c1 += ((c0 >> 51) as u64) as u128;
            a[0] = (c0 as u64) & LOW_51_BIT_MASK;

            c2 += ((c1 >> 51) as u64) as u128;
            a[1] = (c1 as u64) & LOW_51_BIT_MASK;

            c3 += ((c2 >> 51) as u64) as u128;
            a[2] = (c2 as u64) & LOW_51_BIT_MASK;

            c4 += ((c3 >> 51) as u64) as u128;
            a[3] = (c3 as u64) & LOW_51_BIT_MASK;

            let carry: u64 = (c4 >> 51) as u64;
            a[4] = (c4 as u64) & LOW_51_BIT_MASK;

            // To see that this does not overflow, we need a[0] + carry * 19 < 2^64.
            //
            // c4 < a2^2 + 2*a0*a4 + 2*a1*a3 + (carry from c3)
            //    < 2^(102 + 2*b + lg(5)) + 2^64.
            //
            // When b < 3 we get
            //
            // c4 < 2^110.33  so that carry < 2^59.33
            //
            // so that
            //
            // a[0] + carry * 19 < 2^51 + 19 * 2^59.33 < 2^63.58
            //
            // and there is no overflow.
            a[0] += carry * 19;

            // Now a[1] < 2^51 + 2^(64 -51) = 2^51 + 2^13 < 2^(51 + epsilon).
            a[1] += a[0] >> 51;
            a[0] &= LOW_51_BIT_MASK;

            // Now all a[i] < 2^(51 + epsilon) and a = self^(2^k).

            k -= 1;
            if k == 0 {
                break;
            }
        }

        FieldElement51(a)
    }

    /// Returns the square of this field element.
    pub fn square(&self) -> FieldElement51 {
        self.pow2k(1)
    }

    /// Returns 2 times the square of this field element.
    pub fn square2(&self) -> FieldElement51 {
        let mut square = self.pow2k(1);
        for i in 0..5 {
            square.0[i] *= 2;
        }

        square
    }

    #[inline]
    pub(crate) fn batch_mul<const N: usize>(
        a_batch: &[FieldElement51; N],
        b_batch: &[FieldElement51; N],
    ) -> [FieldElement51; N] {
        let mut output = [FieldElement51::ZERO; N];
        batch_mul_dispatch(a_batch.as_slice(), b_batch.as_slice(), output.as_mut_slice());
        output
    }

    #[inline]
    pub(crate) fn batch_square<const N: usize>(a_batch: &[FieldElement51; N]) -> [FieldElement51; N] {
        let mut output = [FieldElement51::ZERO; N];
        batch_square_dispatch(a_batch.as_slice(), output.as_mut_slice());
        output
    }

    /// Batch subtract: batch[i] - target for fixed-size arrays
    #[inline]
    pub(crate) fn batch_sub<const N: usize>(
        batch: &[FieldElement51; N],
        target: &FieldElement51,
    ) -> [FieldElement51; N] {
        let mut output = [FieldElement51::ZERO; N];

        // On non-x86_64 targets, just do scalar.
        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..N {
                output[i] = &batch[i] - target;
            }
            return output;
        }

        // On x86_64, use multiversion dispatch between sse2 baseline and avx2.
        #[cfg(target_arch = "x86_64")]
        {
            batch_sub_dispatch(batch.as_slice(), target, output.as_mut_slice());
            return output;
        }
    }

    /// Batch element-wise vector subtraction: a[i] - b[i] for fixed-size arrays
    #[inline]
    pub(crate) fn batch_vecsub<const N: usize>(
        a_batch: &[FieldElement51; N],
        b_batch: &[FieldElement51; N],
    ) -> [FieldElement51; N] {
        let mut output = [FieldElement51::ZERO; N];
        batch_vecsub_dispatch(a_batch.as_slice(), b_batch.as_slice(), output.as_mut_slice());
        output
    }

    /// Batch add: batch[i] + target for fixed-size arrays
    #[inline]
    pub(crate) fn batch_add<const N: usize>(
        batch: &[FieldElement51; N],
        target: &FieldElement51,
    ) -> [FieldElement51; N] {
        let mut output = [FieldElement51::ZERO; N];
        batch_add_dispatch(batch.as_slice(), target, output.as_mut_slice());
        output
    }

    /// Batch invert (NOT constant-time).
    ///
    /// - If any input is zero: inverts each nonzero individually, sets zeros to ZERO.
    /// - Otherwise: uses a 4-lane batch inversion across the whole batch.
    ///
    /// Big wins come from `Self::batch_mul::<4>()`, which should be multiversioned
    /// to use AVX2 when available.
    #[inline]
    pub(crate) fn batch_invert_not_ct<const BATCH_SIZE: usize>(batch: &mut [Self; BATCH_SIZE]) {
        debug_assert!(BATCH_SIZE % 4 == 0);

        // Non-x86_64 portable fallback: same algorithm, no multiversion.
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::batch_invert_not_ct_core::<BATCH_SIZE>(batch);
            return;
        }

        // x86_64 multiversion dispatch
        #[cfg(target_arch = "x86_64")]
        {
            batch_invert_not_ct_dispatch::<BATCH_SIZE>(batch);
        }
    }

    #[inline(always)]
    fn batch_invert_not_ct_core<const BATCH_SIZE: usize>(batch: &mut [Self; BATCH_SIZE]) {
        debug_assert!(BATCH_SIZE % 4 == 0);

        let any_zero = batch.iter().any(|x| bool::from(x.is_zero()));

        if any_zero {
            // Rare path: safe per-element inversion
            for i in 0..BATCH_SIZE {
                if !bool::from(batch[i].is_zero()) {
                    batch[i] = batch[i].invert();
                } else {
                    batch[i] = Self::ZERO;
                }
            }
            return;
        }

        // Nonzero case: 4-lane batch inversion across the whole batch.
        let num_chunks = BATCH_SIZE / 4;

        // scratch holds the lane-wise prefix products for every element.
        // length = NUM_CHUNKS * 4 == BATCH_SIZE
        let mut scratch = [Self::ONE; BATCH_SIZE];

        // One accumulator per lane
        let mut acc_lanes = [Self::ONE; 4];

        // Forward pass
        for chunk_idx in 0..num_chunks {
            let base = chunk_idx * 4;

            scratch[base]     = acc_lanes[0];
            scratch[base + 1] = acc_lanes[1];
            scratch[base + 2] = acc_lanes[2];
            scratch[base + 3] = acc_lanes[3];

            let input_chunk = [
                batch[base],
                batch[base + 1],
                batch[base + 2],
                batch[base + 3],
            ];

            // SIMD wins happen here (if batch_mul::<4> is multiversioned).
            acc_lanes = Self::batch_mul::<4>(&acc_lanes, &input_chunk);
        }

        // Invert product of lane accumulators (1 inversion total)
        let p01 = &acc_lanes[0] * &acc_lanes[1];
        let p23 = &acc_lanes[2] * &acc_lanes[3];
        let inv_p0123 = (&p01 * &p23).invert();

        // Compute lane-specific factors so we can get inv(acc_lanes[k]) without extra inversions
        let factors = [
            &acc_lanes[1] * &p23, // = acc1 * acc2 * acc3
            &acc_lanes[0] * &p23, // = acc0 * acc2 * acc3
            &p01 * &acc_lanes[3], // = acc0 * acc1 * acc3
            &p01 * &acc_lanes[2], // = acc0 * acc1 * acc2
        ];

        // acc_lanes[k] becomes inv(original acc_lanes[k])
        acc_lanes = Self::batch_mul::<4>(&[inv_p0123; 4], &factors);

        // Reverse pass
        for chunk_idx in (0..num_chunks).rev() {
            let base = chunk_idx * 4;

            // Capture original inputs before overwrite
            let input_chunk = [
                batch[base],
                batch[base + 1],
                batch[base + 2],
                batch[base + 3],
            ];

            let scratch_chunk = [
                scratch[base],
                scratch[base + 1],
                scratch[base + 2],
                scratch[base + 3],
            ];

            // inv(x_i) = inv(prefix_lane) * prefix_before_i
            let results = Self::batch_mul::<4>(&acc_lanes, &scratch_chunk);

            batch[base]     = results[0];
            batch[base + 1] = results[1];
            batch[base + 2] = results[2];
            batch[base + 3] = results[3];

            // Update lane accumulators by multiplying back the original inputs
            acc_lanes = Self::batch_mul::<4>(&acc_lanes, &input_chunk);
        }
    }
}


#[cfg(target_arch = "x86_64")]
#[multiversion(targets("x86_64+sse2", "x86_64+avx2"))]
#[inline]
fn batch_invert_not_ct_dispatch<const BATCH_SIZE: usize>(batch: &mut [FieldElement51; BATCH_SIZE]) {
    FieldElement51::batch_invert_not_ct_core::<BATCH_SIZE>(batch);
}



#[multiversion(targets("x86_64+sse2", "x86_64+avx2"))]
#[inline]
fn batch_add_dispatch(batch: &[FieldElement51], target: &FieldElement51, output: &mut [FieldElement51]) {
    debug_assert_eq!(batch.len(), output.len());

    #[cfg(target_feature = "avx2")]
    {
        batch_add_simd_avx2(batch, target, output);
        return;
    }

    // Scalar fallback (also used for the "default" target)
    for i in 0..batch.len() {
        output[i] = &batch[i] + target;
    }
}

#[cfg(target_feature = "avx2")]
#[inline(always)]
fn batch_add_simd_avx2(batch: &[FieldElement51], target: &FieldElement51, output: &mut [FieldElement51]) {
    use crate::backend::vector::packed_simd::u64x4;

    let n = batch.len();

    let chunks = n / 4;
    let remainder = n % 4;

    // Process 4-element chunks with SIMD
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;

        // Build results directly (no need for the &x - ZERO trick)
        let mut r0 = FieldElement51::ZERO;
        let mut r1 = FieldElement51::ZERO;
        let mut r2 = FieldElement51::ZERO;
        let mut r3 = FieldElement51::ZERO;

        for limb_idx in 0..5 {
            let batch_limbs = u64x4::new(
                batch[base].0[limb_idx],
                batch[base + 1].0[limb_idx],
                batch[base + 2].0[limb_idx],
                batch[base + 3].0[limb_idx],
            );
            let target_limb = u64x4::splat(target.0[limb_idx]);
            let sum = batch_limbs + target_limb;
            let sum_array = sum.to_array();

            r0.0[limb_idx] = sum_array[0];
            r1.0[limb_idx] = sum_array[1];
            r2.0[limb_idx] = sum_array[2];
            r3.0[limb_idx] = sum_array[3];
        }

        output[base] = &r0 + &FieldElement51::ZERO;
        output[base + 1] = &r1 + &FieldElement51::ZERO;
        output[base + 2] = &r2 + &FieldElement51::ZERO;
        output[base + 3] = &r3 + &FieldElement51::ZERO;
    }

    // Process remainder with scalar
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        output[idx] = &batch[idx] + target;
    }
}

#[multiversion(targets("x86_64+sse2", "x86_64+avx2"))]
#[inline]
fn batch_vecsub_dispatch(a_batch: &[FieldElement51], b_batch: &[FieldElement51], output: &mut [FieldElement51]) {
    debug_assert_eq!(a_batch.len(), b_batch.len());
    debug_assert_eq!(a_batch.len(), output.len());

    #[cfg(target_feature = "avx2")]
    {
        batch_vecsub_simd_avx2(a_batch, b_batch, output);
        return;
    }

    // Scalar fallback (also used for the "default" target)
    for i in 0..a_batch.len() {
        output[i] = &a_batch[i] - &b_batch[i];
    }
}

#[cfg(target_feature = "avx2")]
#[inline(always)]
fn batch_vecsub_simd_avx2(a_batch: &[FieldElement51], b_batch: &[FieldElement51], output: &mut [FieldElement51]) {
    use crate::backend::vector::packed_simd::u64x4;

    let n = a_batch.len();

    let chunks = n / 4;
    let remainder = n % 4;

    // Process 4-element chunks with SIMD
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;

        // Build results directly (no need for the &x - ZERO trick)
        let mut r0 = FieldElement51::ZERO;
        let mut r1 = FieldElement51::ZERO;
        let mut r2 = FieldElement51::ZERO;
        let mut r3 = FieldElement51::ZERO;

        for limb_idx in 0..5 {
            let a_limbs = u64x4::new(
                a_batch[base].0[limb_idx],
                a_batch[base + 1].0[limb_idx],
                a_batch[base + 2].0[limb_idx],
                a_batch[base + 3].0[limb_idx],
            );
            let b_limbs = u64x4::new(
                b_batch[base].0[limb_idx],
                b_batch[base + 1].0[limb_idx],
                b_batch[base + 2].0[limb_idx],
                b_batch[base + 3].0[limb_idx],
            );

            // Match your original offsets exactly
            let offset = if limb_idx == 0 {
                u64x4::splat(36028797018963664u64)
            } else {
                u64x4::splat(36028797018963952u64)
            };

            let diff = a_limbs + offset - b_limbs;
            let diff_array = diff.to_array();

            r0.0[limb_idx] = diff_array[0];
            r1.0[limb_idx] = diff_array[1];
            r2.0[limb_idx] = diff_array[2];
            r3.0[limb_idx] = diff_array[3];
        }

        output[base] = &r0 - &FieldElement51::ZERO;
        output[base + 1] = &r1 - &FieldElement51::ZERO;
        output[base + 2] = &r2 - &FieldElement51::ZERO;
        output[base + 3] = &r3 - &FieldElement51::ZERO;
    }

    // Process remainder with scalar
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        output[idx] = &a_batch[idx] - &b_batch[idx];
    }
}

#[multiversion(targets("x86_64+sse2", "x86_64+avx2"))]
#[inline]
fn batch_square_dispatch(a_batch: &[FieldElement51], output: &mut [FieldElement51]) {
    debug_assert_eq!(a_batch.len(), output.len());

    #[cfg(target_feature = "avx2")]
    {
        batch_square_simd_avx2(a_batch, output);
        return;
    }

    // Scalar fallback (also used for the "default" target)
    for i in 0..a_batch.len() {
        output[i] = a_batch[i].square();
    }
}

#[cfg(target_feature = "avx2")]
#[inline(always)]
fn batch_square_simd_avx2(a_batch: &[FieldElement51], output: &mut [FieldElement51]) {
    use crate::backend::vector::packed_simd::u64x4;

    let n = a_batch.len();

    let chunks = n / 4;
    let remainder = n % 4;

    const LOW_51: u64 = (1 << 51) - 1;

    #[inline(always)]
    fn mul64_to_128_simd(a: u64x4, b: u64x4) -> (u64x4, u64x4) {
        let mask_32 = u64x4::splat(0xFFFFFFFF);

        let a_lo = a & mask_32;
        let a_hi = a >> 32;
        let b_lo = b & mask_32;
        let b_hi = b >> 32;

        let lo_lo = a_lo * b_lo;
        let lo_hi = a_lo * b_hi;
        let hi_lo = a_hi * b_lo;
        let hi_hi = a_hi * b_hi;

        let mid = lo_hi + hi_lo;
        let mid_lo = mid << 32;
        let mid_hi = mid >> 32;

        let res_lo: u64x4 = lo_lo + mid_lo;
        let carry = res_lo.cmp_lt(lo_lo).blend(u64x4::splat(1), u64x4::splat(0));
        let res_hi: u64x4 = hi_hi + mid_hi + carry;

        (res_lo, res_hi)
    }

    #[inline(always)]
    fn add_128_simd(a_lo: u64x4, a_hi: u64x4, b_lo: u64x4, b_hi: u64x4) -> (u64x4, u64x4) {
        let sum_lo = a_lo + b_lo;
        let carry = sum_lo.cmp_lt(a_lo).blend(u64x4::splat(1), u64x4::splat(0));
        let sum_hi = a_hi + b_hi + carry;
        (sum_lo, sum_hi)
    }

    #[inline(always)]
    fn double_128_simd(lo: u64x4, hi: u64x4) -> (u64x4, u64x4) {
        let new_hi = (hi << 1) | (lo >> 63);
        let new_lo = lo << 1;
        (new_lo, new_hi)
    }

    // Process 4-element chunks with optimized SIMD
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;

        let factor_19 = u64x4::splat(19);
        let mask = u64x4::splat(LOW_51);

        let mut a = [u64x4::splat(0); 5];
        for i in 0..5 {
            a[i] = u64x4::new(
                a_batch[base].0[i],
                a_batch[base + 1].0[i],
                a_batch[base + 2].0[i],
                a_batch[base + 3].0[i],
            );
        }

        let a3_19 = a[3] * factor_19;
        let a4_19 = a[4] * factor_19;

        let a0_sq = mul64_to_128_simd(a[0], a[0]);
        let a1_sq = mul64_to_128_simd(a[1], a[1]);
        let a2_sq = mul64_to_128_simd(a[2], a[2]);

        let a0_a1 = mul64_to_128_simd(a[0], a[1]);
        let a0_a2 = mul64_to_128_simd(a[0], a[2]);
        let a0_a3 = mul64_to_128_simd(a[0], a[3]);
        let a0_a4 = mul64_to_128_simd(a[0], a[4]);
        let a1_a2 = mul64_to_128_simd(a[1], a[2]);
        let a1_a3 = mul64_to_128_simd(a[1], a[3]);
        let a1_a4_19 = mul64_to_128_simd(a[1], a4_19);
        let a2_a3_19 = mul64_to_128_simd(a[2], a3_19);
        let a2_a4_19 = mul64_to_128_simd(a[2], a4_19);
        let a3_a3_19 = mul64_to_128_simd(a[3], a3_19);
        let a4_a3_19 = mul64_to_128_simd(a[4], a3_19);
        let a4_a4_19 = mul64_to_128_simd(a[4], a4_19);

        let a0_a1_2 = double_128_simd(a0_a1.0, a0_a1.1);
        let a0_a2_2 = double_128_simd(a0_a2.0, a0_a2.1);
        let a0_a3_2 = double_128_simd(a0_a3.0, a0_a3.1);
        let a0_a4_2 = double_128_simd(a0_a4.0, a0_a4.1);
        let a1_a2_2 = double_128_simd(a1_a2.0, a1_a2.1);
        let a1_a3_2 = double_128_simd(a1_a3.0, a1_a3.1);
        let a1_a4_19_2 = double_128_simd(a1_a4_19.0, a1_a4_19.1);
        let a2_a3_19_2 = double_128_simd(a2_a3_19.0, a2_a3_19.1);
        let a2_a4_19_2 = double_128_simd(a2_a4_19.0, a2_a4_19.1);
        let a4_a3_19_2 = double_128_simd(a4_a3_19.0, a4_a3_19.1);

        let (c0_lo, c0_hi) = {
            let (lo, hi) = a0_sq;
            let (lo, hi) = add_128_simd(lo, hi, a1_a4_19_2.0, a1_a4_19_2.1);
            add_128_simd(lo, hi, a2_a3_19_2.0, a2_a3_19_2.1)
        };

        let (c1_lo, c1_hi) = {
            let (lo, hi) = a3_a3_19;
            let (lo, hi) = add_128_simd(lo, hi, a0_a1_2.0, a0_a1_2.1);
            add_128_simd(lo, hi, a2_a4_19_2.0, a2_a4_19_2.1)
        };

        let (c2_lo, c2_hi) = {
            let (lo, hi) = a1_sq;
            let (lo, hi) = add_128_simd(lo, hi, a0_a2_2.0, a0_a2_2.1);
            add_128_simd(lo, hi, a4_a3_19_2.0, a4_a3_19_2.1)
        };

        let (c3_lo, c3_hi) = {
            let (lo, hi) = a4_a4_19;
            let (lo, hi) = add_128_simd(lo, hi, a0_a3_2.0, a0_a3_2.1);
            add_128_simd(lo, hi, a1_a2_2.0, a1_a2_2.1)
        };

        let (c4_lo, c4_hi) = {
            let (lo, hi) = a2_sq;
            let (lo, hi) = add_128_simd(lo, hi, a0_a4_2.0, a0_a4_2.1);
            add_128_simd(lo, hi, a1_a3_2.0, a1_a3_2.1)
        };

        let mut limb0 = c0_lo & mask;
        let mut carry = (c0_hi << 13) | (c0_lo >> 51);

        macro_rules! propagate_carry {
            ($c_lo:expr, $c_hi:expr) => {{
                let acc: u64x4 = $c_lo + carry;
                let limb = acc & mask;
                let mut new_carry = ($c_hi << 13) | (acc >> 51);

                let overflow_mask = acc.cmp_lt($c_lo);
                let overflow_array = overflow_mask.to_array();
                if overflow_array != [0, 0, 0, 0] {
                    new_carry =
                        new_carry + overflow_mask.blend(u64x4::splat(1 << 13), u64x4::splat(0));
                }
                carry = new_carry;
                limb
            }};
        }

        let limb1: u64x4 = propagate_carry!(c1_lo, c1_hi);
        let limb2: u64x4 = propagate_carry!(c2_lo, c2_hi);
        let limb3: u64x4 = propagate_carry!(c3_lo, c3_hi);
        let limb4: u64x4 = propagate_carry!(c4_lo, c4_hi);

        limb0 = limb0 + carry * factor_19;
        let carry5 = limb0 >> 51;
        limb0 = limb0 & mask;
        let limb1: u64x4 = limb1 + carry5;

        let limb0_arr = limb0.to_array();
        let limb1_arr = limb1.to_array();
        let limb2_arr = limb2.to_array();
        let limb3_arr = limb3.to_array();
        let limb4_arr = limb4.to_array();

        output[base] = FieldElement51([
            limb0_arr[0],
            limb1_arr[0],
            limb2_arr[0],
            limb3_arr[0],
            limb4_arr[0],
        ]);
        output[base + 1] = FieldElement51([
            limb0_arr[1],
            limb1_arr[1],
            limb2_arr[1],
            limb3_arr[1],
            limb4_arr[1],
        ]);
        output[base + 2] = FieldElement51([
            limb0_arr[2],
            limb1_arr[2],
            limb2_arr[2],
            limb3_arr[2],
            limb4_arr[2],
        ]);
        output[base + 3] = FieldElement51([
            limb0_arr[3],
            limb1_arr[3],
            limb2_arr[3],
            limb3_arr[3],
            limb4_arr[3],
        ]);
    }

    // Process remainder with scalar
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        output[idx] = a_batch[idx].square();
    }
}


#[multiversion(targets("x86_64+sse2", "x86_64+avx2"))]
#[inline]
fn batch_mul_dispatch(a_batch: &[FieldElement51], b_batch: &[FieldElement51], output: &mut [FieldElement51]) {
    debug_assert_eq!(a_batch.len(), b_batch.len());
    debug_assert_eq!(a_batch.len(), output.len());

    #[cfg(target_feature = "avx2")]
    {
        batch_mul_simd_avx2(a_batch, b_batch, output);
        return;
    }

    for i in 0..a_batch.len() {
        output[i] = &a_batch[i] * &b_batch[i];
    }
}

#[cfg(target_feature = "avx2")]
#[inline(always)]
fn batch_mul_simd_avx2(a_batch: &[FieldElement51], b_batch: &[FieldElement51], output: &mut [FieldElement51]) {
    use crate::backend::vector::packed_simd::u64x4;

    let n = a_batch.len();

    let chunks = n / 4;
    let remainder = n % 4;

    const LOW_51: u64 = (1 << 51) - 1;

    #[inline(always)]
    fn mul64_to_128_simd(a: u64x4, b: u64x4) -> (u64x4, u64x4) {
        let mask_32 = u64x4::splat(0xFFFFFFFF);

        let a_lo = a & mask_32;
        let a_hi = a >> 32;
        let b_lo = b & mask_32;
        let b_hi = b >> 32;

        let lo_lo = a_lo * b_lo;
        let lo_hi = a_lo * b_hi;
        let hi_lo = a_hi * b_lo;
        let hi_hi = a_hi * b_hi;

        let mid = lo_hi + hi_lo;
        let mid_lo = mid << 32;
        let mid_hi = mid >> 32;

        let res_lo: u64x4 = lo_lo + mid_lo;
        let carry = res_lo.cmp_lt(lo_lo).blend(u64x4::splat(1), u64x4::splat(0));
        let res_hi: u64x4 = hi_hi + mid_hi + carry;

        (res_lo, res_hi)
    }

    #[inline(always)]
    fn add_128_simd(a_lo: u64x4, a_hi: u64x4, b_lo: u64x4, b_hi: u64x4) -> (u64x4, u64x4) {
        let sum_lo = a_lo + b_lo;
        let carry = sum_lo.cmp_lt(a_lo).blend(u64x4::splat(1), u64x4::splat(0));
        let sum_hi = a_hi + b_hi + carry;
        (sum_lo, sum_hi)
    }

    // Process 4-element chunks with optimized SIMD
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;

        let factor_19 = u64x4::splat(19);
        let mask = u64x4::splat(LOW_51);

        let mut a = [u64x4::splat(0); 5];
        let mut b = [u64x4::splat(0); 5];

        for i in 0..5 {
            a[i] = u64x4::new(
                a_batch[base].0[i], a_batch[base + 1].0[i],
                a_batch[base + 2].0[i], a_batch[base + 3].0[i]
            );
            b[i] = u64x4::new(
                b_batch[base].0[i], b_batch[base + 1].0[i],
                b_batch[base + 2].0[i], b_batch[base + 3].0[i]
            );
        }

        let b1_19 = b[1] * factor_19;
        let b2_19 = b[2] * factor_19;
        let b3_19 = b[3] * factor_19;
        let b4_19 = b[4] * factor_19;

        let a0_b0 = mul64_to_128_simd(a[0], b[0]);
        let a0_b1 = mul64_to_128_simd(a[0], b[1]);
        let a0_b2 = mul64_to_128_simd(a[0], b[2]);
        let a0_b3 = mul64_to_128_simd(a[0], b[3]);
        let a0_b4 = mul64_to_128_simd(a[0], b[4]);

        let a1_b0 = mul64_to_128_simd(a[1], b[0]);
        let a1_b1 = mul64_to_128_simd(a[1], b[1]);
        let a1_b2 = mul64_to_128_simd(a[1], b[2]);
        let a1_b3 = mul64_to_128_simd(a[1], b[3]);
        let a1_b4_19 = mul64_to_128_simd(a[1], b4_19);

        let a2_b0 = mul64_to_128_simd(a[2], b[0]);
        let a2_b1 = mul64_to_128_simd(a[2], b[1]);
        let a2_b2 = mul64_to_128_simd(a[2], b[2]);
        let a2_b3_19 = mul64_to_128_simd(a[2], b3_19);
        let a2_b4_19 = mul64_to_128_simd(a[2], b4_19);

        let a3_b0 = mul64_to_128_simd(a[3], b[0]);
        let a3_b1 = mul64_to_128_simd(a[3], b[1]);
        let a3_b2_19 = mul64_to_128_simd(a[3], b2_19);
        let a3_b3_19 = mul64_to_128_simd(a[3], b3_19);
        let a3_b4_19 = mul64_to_128_simd(a[3], b4_19);

        let a4_b0 = mul64_to_128_simd(a[4], b[0]);
        let a4_b1_19 = mul64_to_128_simd(a[4], b1_19);
        let a4_b2_19 = mul64_to_128_simd(a[4], b2_19);
        let a4_b3_19 = mul64_to_128_simd(a[4], b3_19);
        let a4_b4_19 = mul64_to_128_simd(a[4], b4_19);

        let (c0_lo, c0_hi) = {
            let (lo, hi) = a0_b0;
            let (lo, hi) = add_128_simd(lo, hi, a4_b1_19.0, a4_b1_19.1);
            let (lo, hi) = add_128_simd(lo, hi, a3_b2_19.0, a3_b2_19.1);
            let (lo, hi) = add_128_simd(lo, hi, a2_b3_19.0, a2_b3_19.1);
            add_128_simd(lo, hi, a1_b4_19.0, a1_b4_19.1)
        };

        let (c1_lo, c1_hi) = {
            let (lo, hi) = a1_b0;
            let (lo, hi) = add_128_simd(lo, hi, a0_b1.0, a0_b1.1);
            let (lo, hi) = add_128_simd(lo, hi, a4_b2_19.0, a4_b2_19.1);
            let (lo, hi) = add_128_simd(lo, hi, a3_b3_19.0, a3_b3_19.1);
            add_128_simd(lo, hi, a2_b4_19.0, a2_b4_19.1)
        };

        let (c2_lo, c2_hi) = {
            let (lo, hi) = a2_b0;
            let (lo, hi) = add_128_simd(lo, hi, a1_b1.0, a1_b1.1);
            let (lo, hi) = add_128_simd(lo, hi, a0_b2.0, a0_b2.1);
            let (lo, hi) = add_128_simd(lo, hi, a4_b3_19.0, a4_b3_19.1);
            add_128_simd(lo, hi, a3_b4_19.0, a3_b4_19.1)
        };

        let (c3_lo, c3_hi) = {
            let (lo, hi) = a3_b0;
            let (lo, hi) = add_128_simd(lo, hi, a2_b1.0, a2_b1.1);
            let (lo, hi) = add_128_simd(lo, hi, a1_b2.0, a1_b2.1);
            let (lo, hi) = add_128_simd(lo, hi, a0_b3.0, a0_b3.1);
            add_128_simd(lo, hi, a4_b4_19.0, a4_b4_19.1)
        };

        let (c4_lo, c4_hi) = {
            let (lo, hi) = a4_b0;
            let (lo, hi) = add_128_simd(lo, hi, a3_b1.0, a3_b1.1);
            let (lo, hi) = add_128_simd(lo, hi, a2_b2.0, a2_b2.1);
            let (lo, hi) = add_128_simd(lo, hi, a1_b3.0, a1_b3.1);
            add_128_simd(lo, hi, a0_b4.0, a0_b4.1)
        };

        let mut limb0 = c0_lo & mask;
        let mut carry = (c0_hi << 13) | (c0_lo >> 51);

        macro_rules! propagate_carry {
            ($c_lo:expr, $c_hi:expr) => {{
                let acc: u64x4 = $c_lo + carry;
                let limb = acc & mask;
                let mut new_carry = ($c_hi << 13) | (acc >> 51);

                let overflow_mask = acc.cmp_lt($c_lo);
                let overflow_array = overflow_mask.to_array();
                if overflow_array != [0, 0, 0, 0] {
                    new_carry = new_carry + overflow_mask.blend(u64x4::splat(1 << 13), u64x4::splat(0));
                }
                carry = new_carry;
                limb
            }};
        }

        let limb1: u64x4 = propagate_carry!(c1_lo, c1_hi);
        let limb2: u64x4 = propagate_carry!(c2_lo, c2_hi);
        let limb3: u64x4 = propagate_carry!(c3_lo, c3_hi);
        let limb4: u64x4 = propagate_carry!(c4_lo, c4_hi);

        limb0 = limb0 + carry * factor_19;
        let carry5 = limb0 >> 51;
        limb0 = limb0 & mask;
        let limb1: u64x4 = limb1 + carry5;

        let limb0_arr = limb0.to_array();
        let limb1_arr = limb1.to_array();
        let limb2_arr = limb2.to_array();
        let limb3_arr = limb3.to_array();
        let limb4_arr = limb4.to_array();

        output[base] = FieldElement51([limb0_arr[0], limb1_arr[0], limb2_arr[0], limb3_arr[0], limb4_arr[0]]);
        output[base + 1] = FieldElement51([limb0_arr[1], limb1_arr[1], limb2_arr[1], limb3_arr[1], limb4_arr[1]]);
        output[base + 2] = FieldElement51([limb0_arr[2], limb1_arr[2], limb2_arr[2], limb3_arr[2], limb4_arr[2]]);
        output[base + 3] = FieldElement51([limb0_arr[3], limb1_arr[3], limb2_arr[3], limb3_arr[3], limb4_arr[3]]);
    }

    // Process remainder with scalar
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        output[idx] = &a_batch[idx] * &b_batch[idx];
    }
}

#[cfg(target_arch = "x86_64")]
#[multiversion(targets("x86_64+sse2", "x86_64+avx2"))]
#[inline]
fn batch_sub_dispatch(batch: &[FieldElement51], target: &FieldElement51, output: &mut [FieldElement51]) {
    debug_assert_eq!(batch.len(), output.len());

    #[cfg(target_feature = "avx2")]
    {
        batch_sub_simd_avx2(batch, target, output);
        return;
    }

    for i in 0..batch.len() {
        output[i] = &batch[i] - target;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn batch_sub_simd_avx2(batch: &[FieldElement51], target: &FieldElement51, output: &mut [FieldElement51]) {
    use crate::backend::vector::packed_simd::u64x4;

    let n = batch.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // Process 4-element chunks with SIMD
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;

        // Build results directly (no need for the &x - ZERO trick)
        let mut r0 = FieldElement51::ZERO;
        let mut r1 = FieldElement51::ZERO;
        let mut r2 = FieldElement51::ZERO;
        let mut r3 = FieldElement51::ZERO;

        for limb_idx in 0..5 {
            let batch_limbs = u64x4::new(
                batch[base].0[limb_idx],
                batch[base + 1].0[limb_idx],
                batch[base + 2].0[limb_idx],
                batch[base + 3].0[limb_idx],
            );
            let target_limb = u64x4::splat(target.0[limb_idx]);

            // Match your original offsets exactly
            let offset = if limb_idx == 0 {
                u64x4::splat(36028797018963664u64)
            } else {
                u64x4::splat(36028797018963952u64)
            };

            let diff = batch_limbs + offset - target_limb;
            let d = diff.to_array();

            r0.0[limb_idx] = d[0];
            r1.0[limb_idx] = d[1];
            r2.0[limb_idx] = d[2];
            r3.0[limb_idx] = d[3];
        }

        output[base] = &r0 - &FieldElement51::ZERO;
        output[base + 1] = &r1 - &FieldElement51::ZERO;
        output[base + 2] = &r2 - &FieldElement51::ZERO;
        output[base + 3] = &r3 - &FieldElement51::ZERO;
    }

    // Process remainder with scalar
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        output[idx] = &batch[idx] - target;
    }
}