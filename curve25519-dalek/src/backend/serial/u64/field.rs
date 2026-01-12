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

    /// Batch element-wise vector subtraction: a[i] - b[i] for fixed-size arrays
    #[inline]
    pub(crate) fn batch_vecadd<const N: usize>(
        a_batch: &[FieldElement51; N],
        b_batch: &[FieldElement51; N],
    ) -> [FieldElement51; N] {
        let mut output = [FieldElement51::ZERO; N];
        batch_vecadd_dispatch(a_batch.as_slice(), b_batch.as_slice(), output.as_mut_slice());
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
        batch_invert_not_ct_dispatch::<BATCH_SIZE, true>(batch);
    }

    /// Same as above, but not checked for 0s
    #[inline]
    pub(crate) fn batch_invert_not_ct_unchecked<const BATCH_SIZE: usize>(batch: &mut [Self; BATCH_SIZE]) {
        batch_invert_not_ct_dispatch::<BATCH_SIZE, false>(batch);
    }

    /// SIMD-striped batch inversion with configurable lane width
    #[inline]
    fn batch_invert_with_lanes<const BATCH_SIZE: usize, const LANES: usize, const CHECKED: bool>(
        batch: &mut [Self; BATCH_SIZE]
    ) {
        if CHECKED {
            // Handle zero elements
            let mut zero_mask = [false; BATCH_SIZE];
            let mut any_zero = false;
            for i in 0..BATCH_SIZE {
                if batch[i].is_zero_not_ct() {
                    zero_mask[i] = true;
                    any_zero = true;
                }
            }
            
            if any_zero {
                for i in 0..BATCH_SIZE {
                    batch[i] = if !zero_mask[i] {
                        batch[i].invert()
                    } else {
                        Self::ZERO
                    };
                }
                return;
            }
        }
        // Split into full chunks and remainder
        let num_full_chunks = BATCH_SIZE / LANES;
        let remainder = BATCH_SIZE % LANES;
        
        // Process full chunks with LANES-wide striped inversion
        if num_full_chunks > 0 {
            let full_batch_size = num_full_chunks * LANES;
            Self::batch_invert_striped::<LANES>(
                &mut batch[..full_batch_size]
            );
        }
        
        // Process remainder
        if remainder > 0 {
            Self::batch_invert_remainder::<LANES>(
                &mut batch[num_full_chunks * LANES..],
                remainder
            );
        }
    }

    /// Core SIMD-striped batch inversion
    /// Requires: batch.len() % LANES == 0
    #[inline(always)]
    fn batch_invert_striped<const LANES: usize>(batch: &mut [Self]) {
        let batch_size = batch.len();
        debug_assert_eq!(batch_size % LANES, 0);
        
        let num_chunks = batch_size / LANES;
        
        // Scratch holds the per-lane prefix products for each element
        let mut scratch = vec![Self::ONE; batch_size];
        
        // Lane accumulators: acc[k] = product of all x[i] where i % LANES == k
        let mut acc_lanes = [Self::ONE; LANES];
        
        // Forward pass: build striped prefix products
        for chunk_idx in 0..num_chunks {
            let base = chunk_idx * LANES;
            
            // Store current lane accumulators as prefix products
            for lane in 0..LANES {
                scratch[base + lane] = acc_lanes[lane];
            }
            
            // Load current chunk
            let mut input_chunk = [Self::ZERO; LANES];
            for lane in 0..LANES {
                input_chunk[lane] = batch[base + lane];
            }
            
            // Vectorized multiply: acc_lanes *= input_chunk
            acc_lanes = Self::batch_mul::<LANES>(&acc_lanes, &input_chunk);
        }
        
        // Invert the product of all lane accumulators (1 inversion total)
        let inv_acc_lanes = Self::invert_lane_product::<LANES>(&acc_lanes);
        acc_lanes = inv_acc_lanes;
        
        // Reverse pass: compute individual inverses
        for chunk_idx in (0..num_chunks).rev() {
            let base = chunk_idx * LANES;
            
            // Capture original inputs before overwrite
            let mut input_chunk = [Self::ZERO; LANES];
            let mut scratch_chunk = [Self::ZERO; LANES];
            for lane in 0..LANES {
                input_chunk[lane] = batch[base + lane];
                scratch_chunk[lane] = scratch[base + lane];
            }
            
            // Compute inverses: x[i]⁻¹ = inv(acc_lane[i % LANES]) × scratch[i]
            let results = Self::batch_mul::<LANES>(&acc_lanes, &scratch_chunk);
            
            // Write results
            for lane in 0..LANES {
                batch[base + lane] = results[lane];
            }
            
            // Update lane accumulators for next iteration
            acc_lanes = Self::batch_mul::<LANES>(&acc_lanes, &input_chunk);
        }
    }

    /// Invert the product of lane accumulators using Montgomery's trick
    /// Returns [inv(acc[0]), inv(acc[1]), ..., inv(acc[LANES-1])]
    #[inline(always)]
    fn invert_lane_product<const LANES: usize>(
        acc_lanes: &[Self; LANES]
    ) -> [Self; LANES] {
        // Dispatch to monomorphized implementations
        // The compiler will optimize away the unused branches for each LANES value
        
        if LANES == 1 {
            let mut result = [Self::ZERO; LANES];
            result[0] = acc_lanes[0].invert();
            result
        } else if LANES == 2 {
            let mut result = [Self::ZERO; LANES];
            let prod = &acc_lanes[0] * &acc_lanes[1];
            let inv_prod = prod.invert();
            result[0] = &inv_prod * &acc_lanes[1];
            result[1] = &inv_prod * &acc_lanes[0];
            result
        } else if LANES == 4 {
            let mut result = [Self::ZERO; LANES];
            
            let p01 = &acc_lanes[0] * &acc_lanes[1];
            let p23 = &acc_lanes[2] * &acc_lanes[3];
            let inv_total = (&p01 * &p23).invert();
            
            // factors[k] = product of all acc_lanes except acc_lanes[k]
            let f0 = &acc_lanes[1] * &p23;
            let f1 = &acc_lanes[0] * &p23;
            let f2 = &p01 * &acc_lanes[3];
            let f3 = &p01 * &acc_lanes[2];
            
            result[0] = &inv_total * &f0;
            result[1] = &inv_total * &f1;
            result[2] = &inv_total * &f2;
            result[3] = &inv_total * &f3;
            result
        } else if LANES == 8 {
            let mut result = [Self::ZERO; LANES];
            
            let p01 = &acc_lanes[0] * &acc_lanes[1];
            let p23 = &acc_lanes[2] * &acc_lanes[3];
            let p45 = &acc_lanes[4] * &acc_lanes[5];
            let p67 = &acc_lanes[6] * &acc_lanes[7];
            
            let p0123 = &p01 * &p23;
            let p4567 = &p45 * &p67;
            
            let inv_total = (&p0123 * &p4567).invert();
            
            // Precompute common subexpressions
            let p23_p4567 = &p23 * &p4567;
            let p01_p4567 = &p01 * &p4567;
            let p67_p0123 = &p67 * &p0123;
            let p45_p0123 = &p45 * &p0123;
            
            result[0] = &inv_total * &(&acc_lanes[1] * &p23_p4567);
            result[1] = &inv_total * &(&acc_lanes[0] * &p23_p4567);
            result[2] = &inv_total * &(&acc_lanes[3] * &p01_p4567);
            result[3] = &inv_total * &(&acc_lanes[2] * &p01_p4567);
            result[4] = &inv_total * &(&acc_lanes[5] * &p67_p0123);
            result[5] = &inv_total * &(&acc_lanes[4] * &p67_p0123);
            result[6] = &inv_total * &(&acc_lanes[7] * &p45_p0123);
            result[7] = &inv_total * &(&acc_lanes[6] * &p45_p0123);
            result
        } else {
            // Generic path for arbitrary LANES
            let mut prefix = [Self::ONE; LANES];
            let mut prod = Self::ONE;
            for i in 0..LANES {
                prefix[i] = prod;
                prod = &prod * &acc_lanes[i];
            }
            
            let inv_total = prod.invert();
            let mut result = [Self::ZERO; LANES];
            let mut suffix = inv_total;
            
            for i in (0..LANES).rev() {
                result[i] = &prefix[i] * &suffix;
                suffix = &suffix * &acc_lanes[i];
            }
            
            result
        }
    }

    /// Handle remainder elements that don't fill a complete SIMD lane
    #[inline(always)]
    fn batch_invert_remainder<const LANES: usize>(
        remainder_batch: &mut [Self],
        count: usize
    ) {
        debug_assert!(count > 0 && count < LANES);
        
        match count {
            1 => {
                remainder_batch[0] = remainder_batch[0].invert();
            }
            2 if LANES >= 2 => {
                // Use 2-lane striped inversion
                let mut pair = [remainder_batch[0], remainder_batch[1]];
                Self::batch_invert_striped::<2>(&mut pair);
                remainder_batch[0] = pair[0];
                remainder_batch[1] = pair[1];
            }
            3 if LANES == 4 => {
                // Pad with ONE (identity element) to fill 4-lane chunk
                let mut padded = [
                    remainder_batch[0],
                    remainder_batch[1],
                    remainder_batch[2],
                    Self::ONE,
                ];
                Self::batch_invert_striped::<4>(&mut padded);
                remainder_batch[0] = padded[0];
                remainder_batch[1] = padded[1];
                remainder_batch[2] = padded[2];
                // padded[3] should be ONE⁻¹ = ONE, we discard it
            }
            _ => {
                // Scalar fallback for other cases
                for i in 0..count {
                    remainder_batch[i] = remainder_batch[i].invert();
                }
            }
        }
    }
}

#[multiversion(targets("x86_64+sse2", "x86_64+avx2", "aarch64+neon"))]
#[inline]
fn batch_invert_not_ct_dispatch<const BATCH_SIZE: usize, const CHECKED: bool>(batch: &mut [FieldElement51; BATCH_SIZE]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && BATCH_SIZE >= 4 {
            FieldElement51::batch_invert_with_lanes::<BATCH_SIZE, 4, CHECKED>(batch);
            return;
        }
        if BATCH_SIZE >= 2 {
            FieldElement51::batch_invert_with_lanes::<BATCH_SIZE, 2, CHECKED>(batch);
            return;
        }
    }
    
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if BATCH_SIZE >= 2 {
            FieldElement51::batch_invert_with_lanes::<BATCH_SIZE, 2, CHECKED>(batch);
            return;
        }
    }
    
    // Scalar fallback or single element
    if BATCH_SIZE == 1  && !batch[0].is_zero_not_ct() {
        batch[0] = batch[0].invert();
    } else {
        FieldElement51::batch_invert_with_lanes::<BATCH_SIZE, 2, CHECKED>(batch);
    }
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

        output[base] = r0;
        output[base + 1] = r1;
        output[base + 2] = r2;
        output[base + 3] = r3;
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
fn batch_vecadd_dispatch(a_batch: &[FieldElement51], b_batch: &[FieldElement51], output: &mut [FieldElement51]) {
    debug_assert_eq!(a_batch.len(), b_batch.len());
    debug_assert_eq!(a_batch.len(), output.len());

    #[cfg(target_feature = "avx2")]
    {
        batch_vecadd_simd_avx2(a_batch, b_batch, output);
        return;
    }

    // Scalar fallback (also used for the "default" target)
    for i in 0..a_batch.len() {
        output[i] = &a_batch[i] + &b_batch[i];
    }
}

#[cfg(target_feature = "avx2")]
#[inline(always)]
fn batch_vecadd_simd_avx2(a_batch: &[FieldElement51], b_batch: &[FieldElement51], output: &mut [FieldElement51]) {
    use crate::backend::vector::packed_simd::u64x4;

    let n = a_batch.len();

    let chunks = n / 4;
    let remainder = n % 4;

    // Process 4-element chunks with SIMD
    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;

        // Build results directly
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

            // Simple SIMD addition
            let sum = a_limbs + b_limbs;
            let sum_array = sum.to_array();

            r0.0[limb_idx] = sum_array[0];
            r1.0[limb_idx] = sum_array[1];
            r2.0[limb_idx] = sum_array[2];
            r3.0[limb_idx] = sum_array[3];
        }

        // No reduction, to match the existing avx2 backend
        output[base] = r0;
        output[base + 1] = r1;
        output[base + 2] = r2;
        output[base + 3] = r3;
    }

    // Process remainder with scalar
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        output[idx] = &a_batch[idx] + &b_batch[idx];
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to generate random field elements for testing
    fn random_field_elements<const N: usize>() -> [FieldElement51; N] {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut result = [FieldElement51::ZERO; N];
        for i in 0..N {
            // Generate random bytes and reduce to field element
            let mut bytes = [0u8; 32];
            rng.fill(&mut bytes);
            result[i] = FieldElement51::from_bytes(&bytes);
        }
        result
    }

    /// Helper to verify batch inversion correctness
    fn verify_batch_inversion<const N: usize>(batch: &[FieldElement51; N]) {
        let mut test_batch = *batch;
        FieldElement51::batch_invert_not_ct(&mut test_batch);
        
        for i in 0..N {
            if batch[i] == FieldElement51::ZERO {
                assert_eq!(test_batch[i], FieldElement51::ZERO, "Zero should map to zero");
            } else {
                let product = &batch[i] * &test_batch[i];
                assert_eq!(product, FieldElement51::ONE, 
                    "x[{}] * inv(x[{}]) should equal ONE", i, i);
            }
        }
    }

    #[test]
    fn test_batch_invert_single() {
        let batch = random_field_elements::<1>();
        verify_batch_inversion(&batch);
    }

    #[test]
    fn test_batch_invert_2_lane() {
        // Test exact 2-lane batch
        let batch = random_field_elements::<2>();
        verify_batch_inversion(&batch);
        
        // Test multiple of 2
        let batch = random_field_elements::<8>();
        verify_batch_inversion(&batch);
    }

    #[test]
    fn test_batch_invert_4_lane() {
        // Test exact 4-lane batch
        let batch = random_field_elements::<4>();
        verify_batch_inversion(&batch);
        
        // Test multiple of 4
        let batch = random_field_elements::<16>();
        verify_batch_inversion(&batch);
    }

    #[test]
    fn test_batch_invert_8_lane() {
        // Test exact 8-lane batch (for future AVX-512)
        let batch = random_field_elements::<8>();
        verify_batch_inversion(&batch);
        
        // Test multiple of 8
        let batch = random_field_elements::<32>();
        verify_batch_inversion(&batch);
    }

    #[test]
    fn test_batch_invert_remainders() {
        // Test various remainder cases
        verify_batch_inversion(&random_field_elements::<3>());
        verify_batch_inversion(&random_field_elements::<5>());
        verify_batch_inversion(&random_field_elements::<6>());
        verify_batch_inversion(&random_field_elements::<7>());
        verify_batch_inversion(&random_field_elements::<9>());
        verify_batch_inversion(&random_field_elements::<11>());
    }

    #[test]
    fn test_batch_invert_large() {
        // Test large batches
        verify_batch_inversion(&random_field_elements::<64>());
        verify_batch_inversion(&random_field_elements::<128>());
    }

    #[test]
    fn test_batch_invert_with_zeros() {
        let mut batch = random_field_elements::<8>();
        batch[2] = FieldElement51::ZERO;
        batch[5] = FieldElement51::ZERO;
        
        verify_batch_inversion(&batch);
    }

    #[test]
    fn test_batch_invert_all_zeros() {
        let batch = [FieldElement51::ZERO; 8];
        let mut test_batch = batch;
        FieldElement51::batch_invert_not_ct(&mut test_batch);
        
        for i in 0..8 {
            assert_eq!(test_batch[i], FieldElement51::ZERO);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Force-specific SIMD width tests
    // ─────────────────────────────────────────────────────────────

    #[test]
    fn test_force_2_lane_inversion() {
        let batch = random_field_elements::<16>();
        let mut test_batch = batch;
        
        // Directly call the 2-lane implementation
        FieldElement51::batch_invert_with_lanes::<16, 2>(&mut test_batch);
        
        for i in 0..16 {
            let product = &batch[i] * &test_batch[i];
            assert_eq!(product, FieldElement51::ONE);
        }
    }

    #[test]
    fn test_force_4_lane_inversion() {
        let batch = random_field_elements::<16>();
        let mut test_batch = batch;
        
        // Directly call the 4-lane implementation
        FieldElement51::batch_invert_with_lanes::<16, 4>(&mut test_batch);
        
        for i in 0..16 {
            let product = &batch[i] * &test_batch[i];
            assert_eq!(product, FieldElement51::ONE);
        }
    }

    #[test]
    fn test_force_8_lane_inversion() {
        let batch = random_field_elements::<16>();
        let mut test_batch = batch;
        
        // Directly call the 8-lane implementation
        FieldElement51::batch_invert_with_lanes::<16, 8>(&mut test_batch);
        
        for i in 0..16 {
            let product = &batch[i] * &test_batch[i];
            assert_eq!(product, FieldElement51::ONE);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Test invert_lane_product in isolation
    // ─────────────────────────────────────────────────────────────

    #[test]
    fn test_invert_lane_product_2() {
        let acc_lanes = random_field_elements::<2>();
        let inv_lanes = FieldElement51::invert_lane_product::<2>(&acc_lanes);
        
        for i in 0..2 {
            let product = &acc_lanes[i] * &inv_lanes[i];
            assert_eq!(product, FieldElement51::ONE);
        }
    }

    #[test]
    fn test_invert_lane_product_4() {
        let acc_lanes = random_field_elements::<4>();
        let inv_lanes = FieldElement51::invert_lane_product::<4>(&acc_lanes);
        
        for i in 0..4 {
            let product = &acc_lanes[i] * &inv_lanes[i];
            assert_eq!(product, FieldElement51::ONE);
        }
    }

    #[test]
    fn test_invert_lane_product_8() {
        let acc_lanes = random_field_elements::<8>();
        let inv_lanes = FieldElement51::invert_lane_product::<8>(&acc_lanes);
        
        for i in 0..8 {
            let product = &acc_lanes[i] * &inv_lanes[i];
            assert_eq!(product, FieldElement51::ONE);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Consistency tests: compare different SIMD widths
    // ─────────────────────────────────────────────────────────────

    #[test]
    fn test_simd_width_consistency() {
        let batch = random_field_elements::<16>();
        
        let mut batch_2lane = batch;
        let mut batch_4lane = batch;
        
        FieldElement51::batch_invert_with_lanes::<16, 2>(&mut batch_2lane);
        FieldElement51::batch_invert_with_lanes::<16, 4>(&mut batch_4lane);
        
        for i in 0..16 {
            assert_eq!(batch_2lane[i], batch_4lane[i],
                "2-lane and 4-lane should produce identical results at index {}", i);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Benchmark helpers (not actual benchmarks, just perf indicators)
    // ─────────────────────────────────────────────────────────────

    #[test]
    #[ignore] // Run with --ignored for perf testing
    fn perf_scalar_vs_2lane() {
        use std::time::Instant;
        
        let batch = random_field_elements::<1000>();
        
        // Scalar (via 1-lane)
        let mut batch_scalar = batch;
        let start = Instant::now();
        for elem in &mut batch_scalar {
            *elem = elem.invert();
        }
        let scalar_time = start.elapsed();
        
        // 2-lane batched
        let mut batch_2lane = batch;
        let start = Instant::now();
        FieldElement51::batch_invert_with_lanes::<1000, 2>(&mut batch_2lane);
        let batch_time = start.elapsed();
        
        println!("Scalar: {:?}, 2-lane batch: {:?}, speedup: {:.2}x",
            scalar_time, batch_time,
            scalar_time.as_secs_f64() / batch_time.as_secs_f64());
    }

    #[test]
    #[ignore] // Run with --ignored for perf testing
    fn perf_scalar_vs_batched() {
        use std::hint::black_box;
        use std::time::Instant;
        
        const BATCH_SIZE: usize = 1000;
        const ITERATIONS: usize = 100;
        
        let batch = random_field_elements::<BATCH_SIZE>();
        
        // ─────────────────────────────────────────────────────────
        // Scalar: individual inversions
        // ─────────────────────────────────────────────────────────
        let mut scalar_total = std::time::Duration::ZERO;
        for _ in 0..ITERATIONS {
            let mut batch_scalar = black_box(batch);
            let start = Instant::now();
            for elem in &mut batch_scalar {
                *elem = black_box(*elem).invert();
            }
            black_box(&batch_scalar);
            scalar_total += start.elapsed();
        }
        let scalar_avg = scalar_total / ITERATIONS as u32;
        
        // ─────────────────────────────────────────────────────────
        // 2-lane batched
        // ─────────────────────────────────────────────────────────
        let mut lane2_total = std::time::Duration::ZERO;
        for _ in 0..ITERATIONS {
            let mut batch_2lane = black_box(batch);
            let start = Instant::now();
            FieldElement51::batch_invert_with_lanes::<BATCH_SIZE, 2>(
                black_box(&mut batch_2lane)
            );
            black_box(&batch_2lane);
            lane2_total += start.elapsed();
        }
        let lane2_avg = lane2_total / ITERATIONS as u32;
        
        // ─────────────────────────────────────────────────────────
        // 4-lane batched
        // ─────────────────────────────────────────────────────────
        let mut lane4_total = std::time::Duration::ZERO;
        for _ in 0..ITERATIONS {
            let mut batch_4lane = black_box(batch);
            let start = Instant::now();
            FieldElement51::batch_invert_with_lanes::<BATCH_SIZE, 4>(
                black_box(&mut batch_4lane)
            );
            black_box(&batch_4lane);
            lane4_total += start.elapsed();
        }
        let lane4_avg = lane4_total / ITERATIONS as u32;
        
        // ─────────────────────────────────────────────────────────
        // 8-lane batched (for AVX-512 or future)
        // ─────────────────────────────────────────────────────────
        let mut lane8_total = std::time::Duration::ZERO;
        for _ in 0..ITERATIONS {
            let mut batch_8lane = black_box(batch);
            let start = Instant::now();
            FieldElement51::batch_invert_with_lanes::<BATCH_SIZE, 8>(
                black_box(&mut batch_8lane)
            );
            black_box(&batch_8lane);
            lane8_total += start.elapsed();
        }
        let lane8_avg = lane8_total / ITERATIONS as u32;
        
        // ─────────────────────────────────────────────────────────
        // Auto-dispatch (uses best available SIMD)
        // ─────────────────────────────────────────────────────────
        let mut auto_total = std::time::Duration::ZERO;
        for _ in 0..ITERATIONS {
            let mut batch_auto = black_box(batch);
            let start = Instant::now();
            FieldElement51::batch_invert_not_ct(black_box(&mut batch_auto));
            black_box(&batch_auto);
            auto_total += start.elapsed();
        }
        let auto_avg = auto_total / ITERATIONS as u32;
        
        println!("\n═══════════════════════════════════════════════════════════");
        println!("Batch Inversion Performance ({} elements, {} iterations)", BATCH_SIZE, ITERATIONS);
        println!("═══════════════════════════════════════════════════════════");
        println!("Scalar (individual):  {:>12?}", scalar_avg);
        println!("2-lane batched:       {:>12?}  ({:.2}x vs scalar)", lane2_avg, scalar_avg.as_secs_f64() / lane2_avg.as_secs_f64());
        println!("4-lane batched:       {:>12?}  ({:.2}x vs scalar)", lane4_avg, scalar_avg.as_secs_f64() / lane4_avg.as_secs_f64());
        println!("8-lane batched:       {:>12?}  ({:.2}x vs scalar)", lane8_avg, scalar_avg.as_secs_f64() / lane8_avg.as_secs_f64());
        println!("Auto-dispatch:        {:>12?}  ({:.2}x vs scalar)", auto_avg, scalar_avg.as_secs_f64() / auto_avg.as_secs_f64());
        println!("───────────────────────────────────────────────────────────");
        
        #[cfg(target_arch = "x86_64")]
        {
            println!("CPU Features: SSE2={}, AVX2={}, AVX512F={}",
                is_x86_feature_detected!("sse2"),
                is_x86_feature_detected!("avx2"),
                is_x86_feature_detected!("avx512f"));
        }
        println!();
    }

    #[test]
    #[ignore]
    fn perf_scaling_by_batch_size() {
        use std::hint::black_box;
        use std::time::Instant;
        
        const ITERATIONS: usize = 50;
        
        println!("\n═══════════════════════════════════════════════════════════");
        println!("Scaling by Batch Size (auto-dispatch, {} iterations)", ITERATIONS);
        println!("═══════════════════════════════════════════════════════════");
        println!("{:>8} {:>12} {:>12} {:>12}", "Size", "Total", "Per-elem", "vs N=8");
        println!("───────────────────────────────────────────────────────────");
        
        // Baseline: 8 elements
        let baseline_per_elem = {
            let batch = random_field_elements::<8>();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let mut b = black_box(batch);
                let start = Instant::now();
                FieldElement51::batch_invert_not_ct(black_box(&mut b));
                black_box(&b);
                total += start.elapsed();
            }
            let avg = total / ITERATIONS as u32;
            let per_elem = avg / 8;
            println!("{:>8} {:>12?} {:>12?} {:>12}", 8, avg, per_elem, "baseline");
            per_elem
        };
        
        macro_rules! bench_size {
            ($size:expr) => {{
                let batch = random_field_elements::<$size>();
                let mut total = std::time::Duration::ZERO;
                for _ in 0..ITERATIONS {
                    let mut b = black_box(batch);
                    let start = Instant::now();
                    FieldElement51::batch_invert_not_ct(black_box(&mut b));
                    black_box(&b);
                    total += start.elapsed();
                }
                let avg = total / ITERATIONS as u32;
                let per_elem = avg / $size as u32;
                let speedup = baseline_per_elem.as_secs_f64() / per_elem.as_secs_f64();
                println!("{:>8} {:>12?} {:>12?} {:>11.2}x", $size, avg, per_elem, speedup);
            }};
        }
        
        bench_size!(16);
        bench_size!(32);
        bench_size!(64);
        bench_size!(128);
        bench_size!(256);
        bench_size!(512);
        bench_size!(1024);
        
        println!();
    }

    #[test]
    #[ignore]
    fn perf_lane_width_comparison() {
        use std::hint::black_box;
        use std::time::Instant;
        
        const BATCH_SIZE: usize = 256;
        const ITERATIONS: usize = 100;
        
        let batch = random_field_elements::<BATCH_SIZE>();
        
        println!("\n═══════════════════════════════════════════════════════════");
        println!("Lane Width Comparison ({} elements, {} iterations)", BATCH_SIZE, ITERATIONS);
        println!("═══════════════════════════════════════════════════════════");
        
        // 1-lane (effectively Montgomery batch, no SIMD striping benefit)
        let lane1_avg = {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let mut b = black_box(batch);
                let start = Instant::now();
                FieldElement51::batch_invert_with_lanes::<BATCH_SIZE, 1>(black_box(&mut b));
                black_box(&b);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        let lane2_avg = {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let mut b = black_box(batch);
                let start = Instant::now();
                FieldElement51::batch_invert_with_lanes::<BATCH_SIZE, 2>(black_box(&mut b));
                black_box(&b);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        let lane4_avg = {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let mut b = black_box(batch);
                let start = Instant::now();
                FieldElement51::batch_invert_with_lanes::<BATCH_SIZE, 4>(black_box(&mut b));
                black_box(&b);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        let lane8_avg = {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let mut b = black_box(batch);
                let start = Instant::now();
                FieldElement51::batch_invert_with_lanes::<BATCH_SIZE, 8>(black_box(&mut b));
                black_box(&b);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        println!("1-lane:  {:>12?}  (baseline)", lane1_avg);
        println!("2-lane:  {:>12?}  ({:.2}x vs 1-lane)", lane2_avg, lane1_avg.as_secs_f64() / lane2_avg.as_secs_f64());
        println!("4-lane:  {:>12?}  ({:.2}x vs 1-lane)", lane4_avg, lane1_avg.as_secs_f64() / lane4_avg.as_secs_f64());
        println!("8-lane:  {:>12?}  ({:.2}x vs 1-lane)", lane8_avg, lane1_avg.as_secs_f64() / lane8_avg.as_secs_f64());
        println!();
    }

    #[test]
    #[ignore]
    fn perf_invert_lane_product_overhead() {
        use std::hint::black_box;
        use std::time::Instant;
        
        const ITERATIONS: usize = 10000;
        
        println!("\n═══════════════════════════════════════════════════════════");
        println!("invert_lane_product Overhead ({} iterations)", ITERATIONS);
        println!("═══════════════════════════════════════════════════════════");
        
        // Single inversion baseline
        let single_avg = {
            let elem = random_field_elements::<1>()[0];
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let e = black_box(elem);
                let start = Instant::now();
                let inv = black_box(e).invert();
                black_box(inv);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // 2-lane invert_lane_product
        let lane2_avg = {
            let acc = random_field_elements::<2>();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let a = black_box(acc);
                let start = Instant::now();
                let inv = FieldElement51::invert_lane_product::<2>(black_box(&a));
                black_box(inv);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // 4-lane invert_lane_product
        let lane4_avg = {
            let acc = random_field_elements::<4>();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let a = black_box(acc);
                let start = Instant::now();
                let inv = FieldElement51::invert_lane_product::<4>(black_box(&a));
                black_box(inv);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // 8-lane invert_lane_product
        let lane8_avg = {
            let acc = random_field_elements::<8>();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let a = black_box(acc);
                let start = Instant::now();
                let inv = FieldElement51::invert_lane_product::<8>(black_box(&a));
                black_box(inv);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        println!("Single invert():     {:>12?}  (baseline)", single_avg);
        println!("2-lane product:      {:>12?}  ({:.2}x single, amortizes 2 inverts)", 
            lane2_avg, lane2_avg.as_secs_f64() / single_avg.as_secs_f64());
        println!("4-lane product:      {:>12?}  ({:.2}x single, amortizes 4 inverts)", 
            lane4_avg, lane4_avg.as_secs_f64() / single_avg.as_secs_f64());
        println!("8-lane product:      {:>12?}  ({:.2}x single, amortizes 8 inverts)", 
            lane8_avg, lane8_avg.as_secs_f64() / single_avg.as_secs_f64());
        println!();
        
        // Theoretical vs actual
        println!("Cost-per-inverse:");
        println!("  Single:   {:>12?}", single_avg);
        println!("  2-lane:   {:>12?} (should be ~{:?} ideal)", lane2_avg / 2, single_avg / 2);
        println!("  4-lane:   {:>12?}  (should be ~{:?} ideal)", lane4_avg / 4, single_avg / 4);
        println!("  8-lane:   {:>12?}  (should be ~{:?} ideal)", lane8_avg / 8, single_avg / 8);
        println!();
    }

    #[test]
    #[ignore]
    fn perf_batch_mul_throughput() {
        use std::hint::black_box;
        use std::time::Instant;
        
        const ITERATIONS: usize = 10000;
        
        println!("\n═══════════════════════════════════════════════════════════");
        println!("batch_mul Throughput ({} iterations)", ITERATIONS);
        println!("═══════════════════════════════════════════════════════════");
        
        // Scalar multiply baseline
        let scalar_avg = {
            let a = random_field_elements::<1>()[0];
            let b = random_field_elements::<1>()[0];
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = &aa * &bb;
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // batch_mul::<2>
        let batch2_avg = {
            let a = random_field_elements::<2>();
            let b = random_field_elements::<2>();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = FieldElement51::batch_mul::<2>(&aa, &bb);
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // batch_mul::<4>
        let batch4_avg = {
            let a = random_field_elements::<4>();
            let b = random_field_elements::<4>();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = FieldElement51::batch_mul::<4>(&aa, &bb);
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // batch_mul::<8>
        let batch8_avg = {
            let a = random_field_elements::<8>();
            let b = random_field_elements::<8>();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = FieldElement51::batch_mul::<8>(&aa, &bb);
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        println!("Scalar mul:      {:>12?}  (baseline)", scalar_avg);
        println!("batch_mul::<2>:  {:>12?}  ({:.2} muls/time of 1)", batch2_avg, 2.0 * scalar_avg.as_secs_f64() / batch2_avg.as_secs_f64());
        println!("batch_mul::<4>:  {:>12?}  ({:.2} muls/time of 1)", batch4_avg, 4.0 * scalar_avg.as_secs_f64() / batch4_avg.as_secs_f64());
        println!("batch_mul::<8>:  {:>12?}  ({:.2} muls/time of 1)", batch8_avg, 8.0 * scalar_avg.as_secs_f64() / batch8_avg.as_secs_f64());
        println!();
        
        println!("Per-element cost:");
        println!("  Scalar:       {:>12?}", scalar_avg);
        println!("  batch<2>:     {:>12?}", batch2_avg / 2);
        println!("  batch<4>:     {:>12?}", batch4_avg / 4);
        println!("  batch<8>:     {:>12?}", batch8_avg / 8);
        println!();
    }

    #[test]
    #[ignore]
    fn perf_batch_add_throughput() {
        use std::hint::black_box;
        use std::time::Instant;
        
        const ITERATIONS: usize = 10000;
        
        println!("\n═══════════════════════════════════════════════════════════");
        println!("batch_add Throughput ({} iterations)", ITERATIONS);
        println!("═══════════════════════════════════════════════════════════");
        
        // Scalar addition baseline
        let scalar_avg = {
            let a = random_field_elements::<1>()[0];
            let b = random_field_elements::<1>()[0];
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = &aa + &bb;
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // batch_add::<2>
        let batch2_avg = {
            let a = random_field_elements::<2>();
            let b = random_field_elements::<2>();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = FieldElement51::batch_vecadd::<2>(&aa, &bb);
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // batch_add::<4>
        let batch4_avg = {
            let a = random_field_elements::<4>();
            let b = random_field_elements::<4>();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = FieldElement51::batch_vecadd::<4>(&aa, &bb);
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // batch_add::<8>
        let batch8_avg = {
            let a = random_field_elements::<8>();
            let b = random_field_elements::<8>();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = FieldElement51::batch_vecadd::<8>(&aa, &bb);
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        println!("Scalar add:      {:>12?}  (baseline)", scalar_avg);
        println!("batch_add::<2>:  {:>12?}  ({:.2} adds/time of 1)", batch2_avg, 2.0 * scalar_avg.as_secs_f64() / batch2_avg.as_secs_f64());
        println!("batch_add::<4>:  {:>12?}  ({:.2} adds/time of 1)", batch4_avg, 4.0 * scalar_avg.as_secs_f64() / batch4_avg.as_secs_f64());
        println!("batch_add::<8>:  {:>12?}  ({:.2} adds/time of 1)", batch8_avg, 8.0 * scalar_avg.as_secs_f64() / batch8_avg.as_secs_f64());
        println!();
        
        println!("Per-element cost:");
        println!("  Scalar:       {:>12?}", scalar_avg);
        println!("  batch<2>:     {:>12?}  ({:.2}x speedup)", batch2_avg / 2, scalar_avg.as_secs_f64() / (batch2_avg.as_secs_f64() / 2.0));
        println!("  batch<4>:     {:>12?}  ({:.2}x speedup)", batch4_avg / 4, scalar_avg.as_secs_f64() / (batch4_avg.as_secs_f64() / 4.0));
        println!("  batch<8>:     {:>12?}  ({:.2}x speedup)", batch8_avg / 8, scalar_avg.as_secs_f64() / (batch8_avg.as_secs_f64() / 8.0));
        println!();
    }

    #[test]
    #[ignore]
    fn perf_batch_sub_throughput() {
        use std::hint::black_box;
        use std::time::Instant;
        
        const ITERATIONS: usize = 10000;
        
        println!("\n═══════════════════════════════════════════════════════════");
        println!("batch_sub Throughput ({} iterations)", ITERATIONS);
        println!("═══════════════════════════════════════════════════════════");
        
        // Scalar subtraction baseline
        let scalar_avg = {
            let a = random_field_elements::<1>()[0];
            let b = random_field_elements::<1>()[0];
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = &aa - &bb;
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // batch_sub::<2>
        let batch2_avg = {
            let a = random_field_elements::<2>();
            let b = random_field_elements::<2>();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = FieldElement51::batch_vecsub::<2>(&aa, &bb);
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // batch_sub::<4>
        let batch4_avg = {
            let a = random_field_elements::<4>();
            let b = random_field_elements::<4>();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = FieldElement51::batch_vecsub::<4>(&aa, &bb);
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // batch_sub::<8>
        let batch8_avg = {
            let a = random_field_elements::<8>();
            let b = random_field_elements::<8>();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = FieldElement51::batch_vecsub::<8>(&aa, &bb);
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        println!("Scalar sub:      {:>12?}  (baseline)", scalar_avg);
        println!("batch_sub::<2>:  {:>12?}  ({:.2} subs/time of 1)", batch2_avg, 2.0 * scalar_avg.as_secs_f64() / batch2_avg.as_secs_f64());
        println!("batch_sub::<4>:  {:>12?}  ({:.2} subs/time of 1)", batch4_avg, 4.0 * scalar_avg.as_secs_f64() / batch4_avg.as_secs_f64());
        println!("batch_sub::<8>:  {:>12?}  ({:.2} subs/time of 1)", batch8_avg, 8.0 * scalar_avg.as_secs_f64() / batch8_avg.as_secs_f64());
        println!();
        
        println!("Per-element cost:");
        println!("  Scalar:       {:>12?}", scalar_avg);
        println!("  batch<2>:     {:>12?}  ({:.2}x speedup)", batch2_avg / 2, scalar_avg.as_secs_f64() / (batch2_avg.as_secs_f64() / 2.0));
        println!("  batch<4>:     {:>12?}  ({:.2}x speedup)", batch4_avg / 4, scalar_avg.as_secs_f64() / (batch4_avg.as_secs_f64() / 4.0));
        println!("  batch<8>:     {:>12?}  ({:.2}x speedup)", batch8_avg / 8, scalar_avg.as_secs_f64() / (batch8_avg.as_secs_f64() / 8.0));
        println!();
    }

    #[test]
    #[ignore]
    fn perf_batch_arithmetic_comparison() {
        use std::hint::black_box;
        use std::time::Instant;
        
        const ITERATIONS: usize = 10000;
        const BATCH_SIZE: usize = 4;
        
        println!("\n═══════════════════════════════════════════════════════════");
        println!("Batch Arithmetic Comparison (batch size = {}, {} iterations)", BATCH_SIZE, ITERATIONS);
        println!("═══════════════════════════════════════════════════════════");
        
        let a = random_field_elements::<BATCH_SIZE>();
        let b = random_field_elements::<BATCH_SIZE>();
        
        // Addition
        let add_avg = {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = FieldElement51::batch_vecadd::<BATCH_SIZE>(&aa, &bb);
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // Subtraction
        let sub_avg = {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = FieldElement51::batch_vecsub::<BATCH_SIZE>(&aa, &bb);
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // Multiplication
        let mul_avg = {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let r = FieldElement51::batch_mul::<BATCH_SIZE>(&aa, &bb);
                black_box(r);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        println!("Operation        Total Time    Per-element   Relative");
        println!("───────────────────────────────────────────────────────────");
        println!("batch_vecadd::<{}>:  {:>12?}  {:>12?}  (baseline)", BATCH_SIZE, add_avg, add_avg / BATCH_SIZE as u32);
        println!("batch_vecsub::<{}>:  {:>12?}  {:>12?}  ({:.2}x cost vs add)", BATCH_SIZE, sub_avg, sub_avg / BATCH_SIZE as u32, sub_avg.as_secs_f64() / add_avg.as_secs_f64());
        println!("batch_mul::<{}>:  {:>12?}  {:>12?}  ({:.2}x cost vs add)", BATCH_SIZE, mul_avg, mul_avg / BATCH_SIZE as u32, mul_avg.as_secs_f64() / add_avg.as_secs_f64());
        println!();
    }

    #[test]
    #[ignore]
    fn perf_batch_ops_scaling() {
        use std::hint::black_box;
        use std::time::Instant;
        
        const ITERATIONS: usize = 5000;
        
        println!("\n═══════════════════════════════════════════════════════════");
        println!("Batch Operations Scaling ({} iterations)", ITERATIONS);
        println!("═══════════════════════════════════════════════════════════");
        
        macro_rules! bench_batch_size {
            ($size:expr) => {{
                let a = random_field_elements::<$size>();
                let b = random_field_elements::<$size>();
                
                let add_time = {
                    let mut total = std::time::Duration::ZERO;
                    for _ in 0..ITERATIONS {
                        let (aa, bb) = (black_box(a), black_box(b));
                        let start = Instant::now();
                        let r = FieldElement51::batch_vecadd::<$size>(&aa, &bb);
                        black_box(r);
                        total += start.elapsed();
                    }
                    total / ITERATIONS as u32
                };
                
                let sub_time = {
                    let mut total = std::time::Duration::ZERO;
                    for _ in 0..ITERATIONS {
                        let (aa, bb) = (black_box(a), black_box(b));
                        let start = Instant::now();
                        let r = FieldElement51::batch_vecsub::<$size>(&aa, &bb);
                        black_box(r);
                        total += start.elapsed();
                    }
                    total / ITERATIONS as u32
                };
                
                let mul_time = {
                    let mut total = std::time::Duration::ZERO;
                    for _ in 0..ITERATIONS {
                        let (aa, bb) = (black_box(a), black_box(b));
                        let start = Instant::now();
                        let r = FieldElement51::batch_mul::<$size>(&aa, &bb);
                        black_box(r);
                        total += start.elapsed();
                    }
                    total / ITERATIONS as u32
                };
                
                println!("N={:>3}:  add={:>10?}  sub={:>10?}  mul={:>10?}  (per-elem: {:>10?} {:>10?} {:>10?})",
                    $size,
                    add_time, sub_time, mul_time,
                    add_time / $size as u32,
                    sub_time / $size as u32,
                    mul_time / $size as u32
                );
            }};
        }
        
        bench_batch_size!(1);
        bench_batch_size!(2);
        bench_batch_size!(4);
        bench_batch_size!(8);
        bench_batch_size!(16);
        bench_batch_size!(32);
        bench_batch_size!(64);
        
        println!();
    }

    #[test]
    #[ignore]
    fn perf_mixed_arithmetic_workload() {
        use std::hint::black_box;
        use std::time::Instant;
        
        const ITERATIONS: usize = 1000;
        const BATCH_SIZE: usize = 16;
        
        println!("\n═══════════════════════════════════════════════════════════");
        println!("Mixed Arithmetic Workload (batch size = {}, {} iterations)", BATCH_SIZE, ITERATIONS);
        println!("═══════════════════════════════════════════════════════════");
        println!("Simulates: c = (a + b) * (a - b)");
        println!();
        
        let a = random_field_elements::<BATCH_SIZE>();
        let b = random_field_elements::<BATCH_SIZE>();
        
        // Scalar approach
        let scalar_avg = {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let mut result = [FieldElement51::ZERO; BATCH_SIZE];
                for i in 0..BATCH_SIZE {
                    let sum = &aa[i] + &bb[i];
                    let diff = &aa[i] - &bb[i];
                    result[i] = &sum * &diff;
                }
                black_box(result);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        // Batched approach
        let batched_avg = {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..ITERATIONS {
                let (aa, bb) = (black_box(a), black_box(b));
                let start = Instant::now();
                let sum = FieldElement51::batch_vecadd::<BATCH_SIZE>(&aa, &bb);
                let diff = FieldElement51::batch_vecsub::<BATCH_SIZE>(&aa, &bb);
                let result = FieldElement51::batch_mul::<BATCH_SIZE>(&sum, &diff);
                black_box(result);
                total += start.elapsed();
            }
            total / ITERATIONS as u32
        };
        
        println!("Scalar loop:  {:>12?}  (baseline)", scalar_avg);
        println!("Batched ops:  {:>12?}  ({:.2}x speedup)", batched_avg, scalar_avg.as_secs_f64() / batched_avg.as_secs_f64());
        println!();
    }

    #[test]
    #[ignore]
    fn perf_all_arithmetic_benchmarks() {
        perf_batch_add_throughput();
        perf_batch_sub_throughput();
        perf_batch_mul_throughput();
        perf_batch_arithmetic_comparison();
        perf_batch_ops_scaling();
        perf_mixed_arithmetic_workload();
    }

    #[test]
    #[ignore]
    fn perf_complete_benchmark_suite() {
        // Run all benchmarks
        println!("\n");
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║          COMPLETE BENCHMARK SUITE                         ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
        
        #[cfg(target_arch = "x86_64")]
        {
            println!("\nCPU Features:");
            println!("  SSE2:     {}", is_x86_feature_detected!("sse2"));
            println!("  AVX2:     {}", is_x86_feature_detected!("avx2"));
            println!("  AVX512F:  {}", is_x86_feature_detected!("avx512f"));
        }
        
        // Batch arithmetic
        perf_batch_add_throughput();
        perf_batch_sub_throughput();
        perf_batch_mul_throughput();
        perf_batch_arithmetic_comparison();
        perf_batch_ops_scaling();
        perf_mixed_arithmetic_workload();
        
        // Batch inversion
        perf_scalar_vs_batched();
        perf_scaling_by_batch_size();
        perf_lane_width_comparison();
        perf_invert_lane_product_overhead();
        
        println!("\n");
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║          BENCHMARK SUITE COMPLETE                         ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!();
    }
}