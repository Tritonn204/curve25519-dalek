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

//! Field arithmetic for ARM NEON.
//!
//! This module provides a `FieldElement51x2` type that packs 2 field elements
//! into NEON 128-bit registers for parallel arithmetic.

use core::ops::{Add, Mul, Neg, Sub};

use subtle::Choice;
use subtle::ConditionallySelectable;

use core::arch::aarch64::*;

use crate::backend::serial::u64::field::FieldElement51;

/// A vector of 2 FieldElement51 values stored in a SIMD-friendly transposed layout.
///
/// Each limb is stored as a `uint64x2_t` containing the corresponding limb from
/// both field elements: `limbs[i] = [fe0.limb[i], fe1.limb[i]]`.
#[derive(Clone, Copy, Debug)]
pub struct FieldElement51x2 {
    /// Transposed storage: limbs[i] = uint64x2_t containing [fe0.limb[i], fe1.limb[i]]
    pub(crate) limbs: [uint64x2_t; 5],
}

/// Specifies which lanes to operate on in 2-wide operations.
#[derive(Copy, Clone, Debug)]
pub enum Lanes {
    /// Lane 0 (first element)
    A,
    /// Lane 1 (second element)
    B,
    /// Both lanes
    AB,
}

impl FieldElement51x2 {
    /// The zero field element (both lanes).
    pub const ZERO: FieldElement51x2 = FieldElement51x2 {
        limbs: unsafe {
            [
                core::mem::transmute([0u64, 0u64]),
                core::mem::transmute([0u64, 0u64]),
                core::mem::transmute([0u64, 0u64]),
                core::mem::transmute([0u64, 0u64]),
                core::mem::transmute([0u64, 0u64]),
            ]
        },
    };

    /// Create a new FieldElement51x2 from 2 FieldElement51 values.
    #[inline(always)]
    pub fn new(a: &FieldElement51, b: &FieldElement51) -> Self {
        unsafe {
            Self {
                limbs: [
                    vld1q_u64([a.0[0], b.0[0]].as_ptr()),
                    vld1q_u64([a.0[1], b.0[1]].as_ptr()),
                    vld1q_u64([a.0[2], b.0[2]].as_ptr()),
                    vld1q_u64([a.0[3], b.0[3]].as_ptr()),
                    vld1q_u64([a.0[4], b.0[4]].as_ptr()),
                ],
            }
        }
    }

    /// Split back into 2 individual FieldElement51 values.
    #[inline(always)]
    pub fn split(&self) -> [FieldElement51; 2] {
        unsafe {
            let mut l0 = [0u64; 2];
            let mut l1 = [0u64; 2];
            let mut l2 = [0u64; 2];
            let mut l3 = [0u64; 2];
            let mut l4 = [0u64; 2];

            vst1q_u64(l0.as_mut_ptr(), self.limbs[0]);
            vst1q_u64(l1.as_mut_ptr(), self.limbs[1]);
            vst1q_u64(l2.as_mut_ptr(), self.limbs[2]);
            vst1q_u64(l3.as_mut_ptr(), self.limbs[3]);
            vst1q_u64(l4.as_mut_ptr(), self.limbs[4]);

            [
                FieldElement51([l0[0], l1[0], l2[0], l3[0], l4[0]]),
                FieldElement51([l0[1], l1[1], l2[1], l3[1], l4[1]]),
            ]
        }
    }

    /// Extract a single lane as a FieldElement51.
    #[inline(always)]
    pub fn extract_lane(&self, lane: usize) -> FieldElement51 {
        debug_assert!(lane < 2);
        let split = self.split();
        split[lane]
    }

    /// Add the two lanes together (returns a scalar FieldElement51).
    #[inline(always)]
    pub fn lanes_sum(&self) -> FieldElement51 {
        let split = self.split();
        &split[0] + &split[1]
    }

    /// Compute the difference and sum of the two lanes: (a - b, a + b).
    #[inline(always)]
    pub fn diff_sum(&self) -> FieldElement51x2 {
        let split = self.split();
        let diff = &split[0] - &split[1];
        let sum = &split[0] + &split[1];
        FieldElement51x2::new(&diff, &sum)
    }

    /// Negate the field elements lazily (without full reduction).
    #[inline(always)]
    pub fn negate_lazy(&self) -> FieldElement51x2 {
        let split = self.split();
        FieldElement51x2::new(&(-&split[0]), &(-&split[1]))
    }

    /// Blend two FieldElement51x2 based on lanes specification.
    #[inline(always)]
    pub fn blend(&self, other: &FieldElement51x2, lanes: Lanes) -> FieldElement51x2 {
        let self_split = self.split();
        let other_split = other.split();

        match lanes {
            Lanes::A => FieldElement51x2::new(&other_split[0], &self_split[1]),
            Lanes::B => FieldElement51x2::new(&self_split[0], &other_split[1]),
            Lanes::AB => *other,
        }
    }

    /// Square both field elements simultaneously.
    #[inline(always)]
    pub fn square(&self) -> FieldElement51x2 {
        let elements = self.split();
        Self::square_batch(&elements)
    }

    #[inline(always)]
    fn mul_scalar_u64x2(v: uint64x2_t, scalar: u64) -> uint64x2_t {
        unsafe {
            let mut arr = [0u64; 2];
            vst1q_u64(arr.as_mut_ptr(), v);
            arr[0] = arr[0].wrapping_mul(scalar);
            arr[1] = arr[1].wrapping_mul(scalar);
            vld1q_u64(arr.as_ptr())
        }
    }

    /// Internal: Square 2 field elements simultaneously.
    #[inline(always)]
    fn square_batch(a_batch: &[FieldElement51; 2]) -> FieldElement51x2 {
        const LOW_51: u64 = (1 << 51) - 1;

        #[inline(always)]
        fn mul64_to_128(a: uint64x2_t, b: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
            unsafe {
                let mut a_arr = [0u64; 2];
                let mut b_arr = [0u64; 2];
                vst1q_u64(a_arr.as_mut_ptr(), a);
                vst1q_u64(b_arr.as_mut_ptr(), b);

                let mut res_lo = [0u64; 2];
                let mut res_hi = [0u64; 2];

                for i in 0..2 {
                    let a_val = a_arr[i];
                    let b_val = b_arr[i];

                    let a_lo = a_val & 0xFFFFFFFF;
                    let a_hi = a_val >> 32;
                    let b_lo = b_val & 0xFFFFFFFF;
                    let b_hi = b_val >> 32;

                    let lo_lo = a_lo * b_lo;
                    let lo_hi = a_lo * b_hi;
                    let hi_lo = a_hi * b_lo;
                    let hi_hi = a_hi * b_hi;

                    let mid = lo_hi.wrapping_add(hi_lo);
                    let mid_lo = mid << 32;
                    let mid_hi = mid >> 32;

                    let temp_lo = lo_lo.wrapping_add(mid_lo);
                    let carry = if temp_lo < lo_lo { 1 } else { 0 };

                    res_lo[i] = temp_lo;
                    res_hi[i] = hi_hi.wrapping_add(mid_hi).wrapping_add(carry);
                }

                (vld1q_u64(res_lo.as_ptr()), vld1q_u64(res_hi.as_ptr()))
            }
        }

        #[inline(always)]
        fn add_128(
            a_lo: uint64x2_t,
            a_hi: uint64x2_t,
            b_lo: uint64x2_t,
            b_hi: uint64x2_t,
        ) -> (uint64x2_t, uint64x2_t) {
            unsafe {
                let sum_lo = vaddq_u64(a_lo, b_lo);
                let carry_mask = vcltq_u64(sum_lo, a_lo);
                let carry = vandq_u64(carry_mask, vdupq_n_u64(1));
                let sum_hi = vaddq_u64(vaddq_u64(a_hi, b_hi), carry);
                (sum_lo, sum_hi)
            }
        }

        #[inline(always)]
        fn double_128(lo: uint64x2_t, hi: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
            unsafe {
                let new_hi = vorrq_u64(vshlq_n_u64(hi, 1), vshrq_n_u64(lo, 63));
                let new_lo = vshlq_n_u64(lo, 1);
                (new_lo, new_hi)
            }
        }

        unsafe {
            let mut a = [vdupq_n_u64(0); 5];
            for i in 0..5 {
                a[i] = vld1q_u64([a_batch[0].0[i], a_batch[1].0[i]].as_ptr());
            }

            let a3_19 = Self::mul_scalar_u64x2(a[3], 19);
            let a4_19 = Self::mul_scalar_u64x2(a[4], 19);

            let a0_sq = mul64_to_128(a[0], a[0]);
            let a1_sq = mul64_to_128(a[1], a[1]);
            let a2_sq = mul64_to_128(a[2], a[2]);

            let a0_a1 = mul64_to_128(a[0], a[1]);
            let a0_a2 = mul64_to_128(a[0], a[2]);
            let a0_a3 = mul64_to_128(a[0], a[3]);
            let a0_a4 = mul64_to_128(a[0], a[4]);
            let a1_a2 = mul64_to_128(a[1], a[2]);
            let a1_a3 = mul64_to_128(a[1], a[3]);
            let a1_a4_19 = mul64_to_128(a[1], a4_19);
            let a2_a3_19 = mul64_to_128(a[2], a3_19);
            let a2_a4_19 = mul64_to_128(a[2], a4_19);
            let a3_a3_19 = mul64_to_128(a[3], a3_19);
            let a4_a3_19 = mul64_to_128(a[4], a3_19);
            let a4_a4_19 = mul64_to_128(a[4], a4_19);

            let a0_a1_2 = double_128(a0_a1.0, a0_a1.1);
            let a0_a2_2 = double_128(a0_a2.0, a0_a2.1);
            let a0_a3_2 = double_128(a0_a3.0, a0_a3.1);
            let a0_a4_2 = double_128(a0_a4.0, a0_a4.1);
            let a1_a2_2 = double_128(a1_a2.0, a1_a2.1);
            let a1_a3_2 = double_128(a1_a3.0, a1_a3.1);
            let a1_a4_19_2 = double_128(a1_a4_19.0, a1_a4_19.1);
            let a2_a3_19_2 = double_128(a2_a3_19.0, a2_a3_19.1);
            let a2_a4_19_2 = double_128(a2_a4_19.0, a2_a4_19.1);
            let a4_a3_19_2 = double_128(a4_a3_19.0, a4_a3_19.1);

            let (c0_lo, c0_hi) = {
                let (lo, hi) = a0_sq;
                let (lo, hi) = add_128(lo, hi, a1_a4_19_2.0, a1_a4_19_2.1);
                add_128(lo, hi, a2_a3_19_2.0, a2_a3_19_2.1)
            };

            let (c1_lo, c1_hi) = {
                let (lo, hi) = a3_a3_19;
                let (lo, hi) = add_128(lo, hi, a0_a1_2.0, a0_a1_2.1);
                add_128(lo, hi, a2_a4_19_2.0, a2_a4_19_2.1)
            };

            let (c2_lo, c2_hi) = {
                let (lo, hi) = a1_sq;
                let (lo, hi) = add_128(lo, hi, a0_a2_2.0, a0_a2_2.1);
                add_128(lo, hi, a4_a3_19_2.0, a4_a3_19_2.1)
            };

            let (c3_lo, c3_hi) = {
                let (lo, hi) = a4_a4_19;
                let (lo, hi) = add_128(lo, hi, a0_a3_2.0, a0_a3_2.1);
                add_128(lo, hi, a1_a2_2.0, a1_a2_2.1)
            };

            let (c4_lo, c4_hi) = {
                let (lo, hi) = a2_sq;
                let (lo, hi) = add_128(lo, hi, a0_a4_2.0, a0_a4_2.1);
                add_128(lo, hi, a1_a3_2.0, a1_a3_2.1)
            };

            let mask_51 = vdupq_n_u64(LOW_51);
            let mut limb0 = vandq_u64(c0_lo, mask_51);
            let mut carry = vorrq_u64(vshlq_n_u64(c0_hi, 13), vshrq_n_u64(c0_lo, 51));

            macro_rules! propagate_carry {
                ($c_lo:expr, $c_hi:expr) => {{
                    let acc = vaddq_u64($c_lo, carry);
                    let limb = vandq_u64(acc, mask_51);
                    let mut new_carry = vorrq_u64(vshlq_n_u64($c_hi, 13), vshrq_n_u64(acc, 51));

                    let overflow_mask = vcltq_u64(acc, $c_lo);
                    let mut overflow_arr = [0u64; 2];
                    vst1q_u64(overflow_arr.as_mut_ptr(), overflow_mask);
                    if overflow_arr != [0, 0] {
                        new_carry =
                            vaddq_u64(new_carry, vandq_u64(overflow_mask, vdupq_n_u64(1 << 13)));
                    }

                    carry = new_carry;
                    limb
                }};
            }

            let limb1 = propagate_carry!(c1_lo, c1_hi);
            let limb2 = propagate_carry!(c2_lo, c2_hi);
            let limb3 = propagate_carry!(c3_lo, c3_hi);
            let limb4 = propagate_carry!(c4_lo, c4_hi);

            limb0 = vaddq_u64(limb0, Self::mul_scalar_u64x2(carry, 19));
            let carry5 = vshrq_n_u64(limb0, 51);
            limb0 = vandq_u64(limb0, mask_51);
            let limb1 = vaddq_u64(limb1, carry5);

            FieldElement51x2 {
                limbs: [limb0, limb1, limb2, limb3, limb4],
            }
        }
    }

    /// Internal: Multiply 2 pairs of field elements simultaneously.
    #[inline(always)]
    fn mul_batch(
        a_batch: &[FieldElement51; 2],
        b_batch: &[FieldElement51; 2],
    ) -> FieldElement51x2 {
        const LOW_51: u64 = (1 << 51) - 1;

        #[inline(always)]
        fn mul64_to_128(a: uint64x2_t, b: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
            unsafe {
                let mut a_arr = [0u64; 2];
                let mut b_arr = [0u64; 2];
                vst1q_u64(a_arr.as_mut_ptr(), a);
                vst1q_u64(b_arr.as_mut_ptr(), b);

                let mut res_lo = [0u64; 2];
                let mut res_hi = [0u64; 2];

                for i in 0..2 {
                    let a_val = a_arr[i];
                    let b_val = b_arr[i];

                    let a_lo = a_val & 0xFFFFFFFF;
                    let a_hi = a_val >> 32;
                    let b_lo = b_val & 0xFFFFFFFF;
                    let b_hi = b_val >> 32;

                    let lo_lo = a_lo * b_lo;
                    let lo_hi = a_lo * b_hi;
                    let hi_lo = a_hi * b_lo;
                    let hi_hi = a_hi * b_hi;

                    let mid = lo_hi.wrapping_add(hi_lo);
                    let mid_lo = mid << 32;
                    let mid_hi = mid >> 32;

                    let temp_lo = lo_lo.wrapping_add(mid_lo);
                    let carry = if temp_lo < lo_lo { 1 } else { 0 };

                    res_lo[i] = temp_lo;
                    res_hi[i] = hi_hi.wrapping_add(mid_hi).wrapping_add(carry);
                }

                (vld1q_u64(res_lo.as_ptr()), vld1q_u64(res_hi.as_ptr()))
            }
        }

        #[inline(always)]
        fn add_128(
            a_lo: uint64x2_t,
            a_hi: uint64x2_t,
            b_lo: uint64x2_t,
            b_hi: uint64x2_t,
        ) -> (uint64x2_t, uint64x2_t) {
            unsafe {
                let sum_lo = vaddq_u64(a_lo, b_lo);
                let carry_mask = vcltq_u64(sum_lo, a_lo);
                let carry = vandq_u64(carry_mask, vdupq_n_u64(1));
                let sum_hi = vaddq_u64(vaddq_u64(a_hi, b_hi), carry);
                (sum_lo, sum_hi)
            }
        }

        unsafe {
            let mut a = [vdupq_n_u64(0); 5];
            let mut b = [vdupq_n_u64(0); 5];

            for i in 0..5 {
                a[i] = vld1q_u64([a_batch[0].0[i], a_batch[1].0[i]].as_ptr());
                b[i] = vld1q_u64([b_batch[0].0[i], b_batch[1].0[i]].as_ptr());
            }

            let b1_19 = Self::mul_scalar_u64x2(b[1], 19);
            let b2_19 = Self::mul_scalar_u64x2(b[2], 19);
            let b3_19 = Self::mul_scalar_u64x2(b[3], 19);
            let b4_19 = Self::mul_scalar_u64x2(b[4], 19);

            let a0_b0 = mul64_to_128(a[0], b[0]);
            let a0_b1 = mul64_to_128(a[0], b[1]);
            let a0_b2 = mul64_to_128(a[0], b[2]);
            let a0_b3 = mul64_to_128(a[0], b[3]);
            let a0_b4 = mul64_to_128(a[0], b[4]);

            let a1_b0 = mul64_to_128(a[1], b[0]);
            let a1_b1 = mul64_to_128(a[1], b[1]);
            let a1_b2 = mul64_to_128(a[1], b[2]);
            let a1_b3 = mul64_to_128(a[1], b[3]);
            let a1_b4_19 = mul64_to_128(a[1], b4_19);

            let a2_b0 = mul64_to_128(a[2], b[0]);
            let a2_b1 = mul64_to_128(a[2], b[1]);
            let a2_b2 = mul64_to_128(a[2], b[2]);
            let a2_b3_19 = mul64_to_128(a[2], b3_19);
            let a2_b4_19 = mul64_to_128(a[2], b4_19);

            let a3_b0 = mul64_to_128(a[3], b[0]);
            let a3_b1 = mul64_to_128(a[3], b[1]);
            let a3_b2_19 = mul64_to_128(a[3], b2_19);
            let a3_b3_19 = mul64_to_128(a[3], b3_19);
            let a3_b4_19 = mul64_to_128(a[3], b4_19);

            let a4_b0 = mul64_to_128(a[4], b[0]);
            let a4_b1_19 = mul64_to_128(a[4], b1_19);
            let a4_b2_19 = mul64_to_128(a[4], b2_19);
            let a4_b3_19 = mul64_to_128(a[4], b3_19);
            let a4_b4_19 = mul64_to_128(a[4], b4_19);

            let (c0_lo, c0_hi) = {
                let (lo, hi) = a0_b0;
                let (lo, hi) = add_128(lo, hi, a4_b1_19.0, a4_b1_19.1);
                let (lo, hi) = add_128(lo, hi, a3_b2_19.0, a3_b2_19.1);
                let (lo, hi) = add_128(lo, hi, a2_b3_19.0, a2_b3_19.1);
                add_128(lo, hi, a1_b4_19.0, a1_b4_19.1)
            };

            let (c1_lo, c1_hi) = {
                let (lo, hi) = a1_b0;
                let (lo, hi) = add_128(lo, hi, a0_b1.0, a0_b1.1);
                let (lo, hi) = add_128(lo, hi, a4_b2_19.0, a4_b2_19.1);
                let (lo, hi) = add_128(lo, hi, a3_b3_19.0, a3_b3_19.1);
                add_128(lo, hi, a2_b4_19.0, a2_b4_19.1)
            };

            let (c2_lo, c2_hi) = {
                let (lo, hi) = a2_b0;
                let (lo, hi) = add_128(lo, hi, a1_b1.0, a1_b1.1);
                let (lo, hi) = add_128(lo, hi, a0_b2.0, a0_b2.1);
                let (lo, hi) = add_128(lo, hi, a4_b3_19.0, a4_b3_19.1);
                add_128(lo, hi, a3_b4_19.0, a3_b4_19.1)
            };

            let (c3_lo, c3_hi) = {
                let (lo, hi) = a3_b0;
                let (lo, hi) = add_128(lo, hi, a2_b1.0, a2_b1.1);
                let (lo, hi) = add_128(lo, hi, a1_b2.0, a1_b2.1);
                let (lo, hi) = add_128(lo, hi, a0_b3.0, a0_b3.1);
                add_128(lo, hi, a4_b4_19.0, a4_b4_19.1)
            };

            let (c4_lo, c4_hi) = {
                let (lo, hi) = a4_b0;
                let (lo, hi) = add_128(lo, hi, a3_b1.0, a3_b1.1);
                let (lo, hi) = add_128(lo, hi, a2_b2.0, a2_b2.1);
                let (lo, hi) = add_128(lo, hi, a1_b3.0, a1_b3.1);
                add_128(lo, hi, a0_b4.0, a0_b4.1)
            };

            let mask_51 = vdupq_n_u64(LOW_51);
            let mut limb0 = vandq_u64(c0_lo, mask_51);
            let mut carry = vorrq_u64(vshlq_n_u64(c0_hi, 13), vshrq_n_u64(c0_lo, 51));

            macro_rules! propagate_carry {
                ($c_lo:expr, $c_hi:expr) => {{
                    let acc = vaddq_u64($c_lo, carry);
                    let limb = vandq_u64(acc, mask_51);
                    let mut new_carry = vorrq_u64(vshlq_n_u64($c_hi, 13), vshrq_n_u64(acc, 51));

                    let overflow_mask = vcltq_u64(acc, $c_lo);
                    let mut overflow_arr = [0u64; 2];
                    vst1q_u64(overflow_arr.as_mut_ptr(), overflow_mask);
                    if overflow_arr != [0, 0] {
                        new_carry =
                            vaddq_u64(new_carry, vandq_u64(overflow_mask, vdupq_n_u64(1 << 13)));
                    }

                    carry = new_carry;
                    limb
                }};
            }

            let limb1 = propagate_carry!(c1_lo, c1_hi);
            let limb2 = propagate_carry!(c2_lo, c2_hi);
            let limb3 = propagate_carry!(c3_lo, c3_hi);
            let limb4 = propagate_carry!(c4_lo, c4_hi);

            limb0 = vaddq_u64(limb0, Self::mul_scalar_u64x2(carry, 19));
            let carry5 = vshrq_n_u64(limb0, 51);
            limb0 = vandq_u64(limb0, mask_51);
            let limb1 = vaddq_u64(limb1, carry5);

            FieldElement51x2 {
                limbs: [limb0, limb1, limb2, limb3, limb4],
            }
        }
    }
}

impl Add<&FieldElement51x2> for &FieldElement51x2 {
    type Output = FieldElement51x2;

    #[inline(always)]
    fn add(self, rhs: &FieldElement51x2) -> FieldElement51x2 {
        unsafe {
            FieldElement51x2 {
                limbs: [
                    vaddq_u64(self.limbs[0], rhs.limbs[0]),
                    vaddq_u64(self.limbs[1], rhs.limbs[1]),
                    vaddq_u64(self.limbs[2], rhs.limbs[2]),
                    vaddq_u64(self.limbs[3], rhs.limbs[3]),
                    vaddq_u64(self.limbs[4], rhs.limbs[4]),
                ],
            }
        }
    }
}

impl Sub<&FieldElement51x2> for &FieldElement51x2 {
    type Output = FieldElement51x2;

    #[inline(always)]
    fn sub(self, rhs: &FieldElement51x2) -> FieldElement51x2 {
        // Add 2*p to avoid underflow before subtraction
        const LOW_51_2P: u64 = (1u64 << 51) - 1 + (1u64 << 51) - 19;
        const LOW_51_2P_MID: u64 = (1u64 << 51) - 1 + (1u64 << 51) - 1;

        unsafe {
            let bias0 = vdupq_n_u64(LOW_51_2P);
            let bias = vdupq_n_u64(LOW_51_2P_MID);

            FieldElement51x2 {
                limbs: [
                    vsubq_u64(vaddq_u64(self.limbs[0], bias0), rhs.limbs[0]),
                    vsubq_u64(vaddq_u64(self.limbs[1], bias), rhs.limbs[1]),
                    vsubq_u64(vaddq_u64(self.limbs[2], bias), rhs.limbs[2]),
                    vsubq_u64(vaddq_u64(self.limbs[3], bias), rhs.limbs[3]),
                    vsubq_u64(vaddq_u64(self.limbs[4], bias), rhs.limbs[4]),
                ],
            }
        }
    }
}

impl Mul<&FieldElement51x2> for &FieldElement51x2 {
    type Output = FieldElement51x2;

    #[inline(always)]
    fn mul(self, rhs: &FieldElement51x2) -> FieldElement51x2 {
        let a = self.split();
        let b = rhs.split();
        FieldElement51x2::mul_batch(&a, &b)
    }
}

impl Neg for &FieldElement51x2 {
    type Output = FieldElement51x2;

    #[inline(always)]
    fn neg(self) -> FieldElement51x2 {
        self.negate_lazy()
    }
}

impl ConditionallySelectable for FieldElement51x2 {
    fn conditional_select(
        a: &FieldElement51x2,
        b: &FieldElement51x2,
        choice: Choice,
    ) -> FieldElement51x2 {
        unsafe {
            let mask = (-(choice.unwrap_u8() as i64)) as u64;
            let mask_vec = vdupq_n_u64(mask);

            FieldElement51x2 {
                limbs: [
                    veorq_u64(
                        a.limbs[0],
                        vandq_u64(mask_vec, veorq_u64(a.limbs[0], b.limbs[0])),
                    ),
                    veorq_u64(
                        a.limbs[1],
                        vandq_u64(mask_vec, veorq_u64(a.limbs[1], b.limbs[1])),
                    ),
                    veorq_u64(
                        a.limbs[2],
                        vandq_u64(mask_vec, veorq_u64(a.limbs[2], b.limbs[2])),
                    ),
                    veorq_u64(
                        a.limbs[3],
                        vandq_u64(mask_vec, veorq_u64(a.limbs[3], b.limbs[3])),
                    ),
                    veorq_u64(
                        a.limbs[4],
                        vandq_u64(mask_vec, veorq_u64(a.limbs[4], b.limbs[4])),
                    ),
                ],
            }
        }
    }

    fn conditional_assign(&mut self, other: &FieldElement51x2, choice: Choice) {
        unsafe {
            let mask = (-(choice.unwrap_u8() as i64)) as u64;
            let mask_vec = vdupq_n_u64(mask);

            self.limbs[0] = veorq_u64(
                self.limbs[0],
                vandq_u64(mask_vec, veorq_u64(self.limbs[0], other.limbs[0])),
            );
            self.limbs[1] = veorq_u64(
                self.limbs[1],
                vandq_u64(mask_vec, veorq_u64(self.limbs[1], other.limbs[1])),
            );
            self.limbs[2] = veorq_u64(
                self.limbs[2],
                vandq_u64(mask_vec, veorq_u64(self.limbs[2], other.limbs[2])),
            );
            self.limbs[3] = veorq_u64(
                self.limbs[3],
                vandq_u64(mask_vec, veorq_u64(self.limbs[3], other.limbs[3])),
            );
            self.limbs[4] = veorq_u64(
                self.limbs[4],
                vandq_u64(mask_vec, veorq_u64(self.limbs[4], other.limbs[4])),
            );
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_field_element_51x2_new_split_roundtrip() {
        let a = FieldElement51([1, 2, 3, 4, 5]);
        let b = FieldElement51([6, 7, 8, 9, 10]);

        let packed = FieldElement51x2::new(&a, &b);
        let [a_out, b_out] = packed.split();

        assert_eq!(a.0, a_out.0);
        assert_eq!(b.0, b_out.0);
    }

    #[test]
    fn test_field_element_51x2_add() {
        let a = FieldElement51([1, 0, 0, 0, 0]);
        let b = FieldElement51([2, 0, 0, 0, 0]);
        let c = FieldElement51([3, 0, 0, 0, 0]);
        let d = FieldElement51([4, 0, 0, 0, 0]);

        let ab = FieldElement51x2::new(&a, &b);
        let cd = FieldElement51x2::new(&c, &d);
        let result = &ab + &cd;
        let [r0, r1] = result.split();

        assert_eq!(r0.0[0], 4); // 1 + 3
        assert_eq!(r1.0[0], 6); // 2 + 4
    }

    #[test]
    fn test_field_element_51x2_mul() {
        use crate::backend::serial::u64::field::FieldElement51;

        let a = FieldElement51::ONE;
        let b = FieldElement51::ONE;

        let packed = FieldElement51x2::new(&a, &b);
        let result = &packed * &packed;
        let [r0, r1] = result.split();

        // 1 * 1 = 1
        assert_eq!(r0, FieldElement51::ONE);
        assert_eq!(r1, FieldElement51::ONE);
    }
}