// -*- mode: rust; -*-
//
// This file is part of curve25519-dalek.
// See LICENSE for licensing information.

#![allow(unused_unsafe)]

//! This module defines wrappers over platform-specific SIMD types to make them
//! more convenient to use.

use core::ops::{Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul, Shl, Shr, Sub};

#[cfg(target_arch = "x86_64")]
use curve25519_dalek_derive::unsafe_target_feature;

// ═══════════════════════════════════════════════════════════════════════════
// Architecture-specific imports
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// ═══════════════════════════════════════════════════════════════════════════
// Macro for x86_64 types (256-bit AVX2)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
macro_rules! impl_shared {
    (
        $ty:ident,
        $lane_ty:ident,
        $add_intrinsic:ident,
        $sub_intrinsic:ident,
        $shl_intrinsic:ident,
        $shr_intrinsic:ident,
        $extract_intrinsic:ident
    ) => {
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Debug)]
        #[repr(transparent)]
        pub struct $ty(__m256i);

        #[unsafe_target_feature("avx2")]
        impl From<$ty> for __m256i {
            #[inline]
            fn from(value: $ty) -> __m256i {
                value.0
            }
        }

        #[unsafe_target_feature("avx2")]
        impl From<__m256i> for $ty {
            #[inline]
            fn from(value: __m256i) -> $ty {
                $ty(value)
            }
        }

        #[unsafe_target_feature("avx2")]
        impl PartialEq for $ty {
            #[inline]
            fn eq(&self, rhs: &$ty) -> bool {
                unsafe {
                    let m = _mm256_cmpeq_epi8(self.0, rhs.0);
                    _mm256_movemask_epi8(m) == -1
                }
            }
        }

        impl Eq for $ty {}

        #[unsafe_target_feature("avx2")]
        impl Add for $ty {
            type Output = Self;
            #[inline]
            fn add(self, rhs: $ty) -> Self {
                unsafe { $add_intrinsic(self.0, rhs.0).into() }
            }
        }

        #[allow(clippy::assign_op_pattern)]
        #[unsafe_target_feature("avx2")]
        impl AddAssign for $ty {
            #[inline]
            fn add_assign(&mut self, rhs: $ty) {
                *self = *self + rhs
            }
        }

        #[unsafe_target_feature("avx2")]
        impl Sub for $ty {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: $ty) -> Self {
                unsafe { $sub_intrinsic(self.0, rhs.0).into() }
            }
        }

        #[unsafe_target_feature("avx2")]
        impl BitAnd for $ty {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: $ty) -> Self {
                unsafe { _mm256_and_si256(self.0, rhs.0).into() }
            }
        }

        #[unsafe_target_feature("avx2")]
        impl BitXor for $ty {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: $ty) -> Self {
                unsafe { _mm256_xor_si256(self.0, rhs.0).into() }
            }
        }

        #[unsafe_target_feature("avx2")]
        impl BitOr for $ty {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: $ty) -> Self {
                unsafe { _mm256_or_si256(self.0, rhs.0).into() }
            }
        }

        #[allow(clippy::assign_op_pattern)]
        #[unsafe_target_feature("avx2")]
        impl BitAndAssign for $ty {
            #[inline]
            fn bitand_assign(&mut self, rhs: $ty) {
                *self = *self & rhs;
            }
        }

        #[allow(clippy::assign_op_pattern)]
        #[unsafe_target_feature("avx2")]
        impl BitXorAssign for $ty {
            #[inline]
            fn bitxor_assign(&mut self, rhs: $ty) {
                *self = *self ^ rhs;
            }
        }

        #[allow(clippy::assign_op_pattern)]
        #[unsafe_target_feature("avx2")]
        impl BitOrAssign for $ty {
            #[inline]
            fn bitor_assign(&mut self, rhs: $ty) {
                *self = *self | rhs;
            }
        }

        #[unsafe_target_feature("avx2")]
        #[allow(dead_code)]
        impl $ty {
            #[inline]
            pub fn shl<const N: i32>(self) -> Self {
                unsafe { $shl_intrinsic(self.0, N).into() }
            }

            #[inline]
            pub fn shr<const N: i32>(self) -> Self {
                unsafe { $shr_intrinsic(self.0, N).into() }
            }

            #[inline]
            pub fn extract<const N: i32>(self) -> $lane_ty {
                unsafe { $extract_intrinsic(self.0, N) as $lane_ty }
            }
        }
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// Macro for aarch64 types (128-bit NEON, emulate 256-bit with pairs)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
macro_rules! impl_shared {
    (
        $ty:ident,
        $lane_ty:ident,
        $neon_ty:ty,
        $add_intrinsic:ident,
        $sub_intrinsic:ident,
        $shl_intrinsic:ident,
        $shr_intrinsic:ident,
        $and_intrinsic:ident,
        $xor_intrinsic:ident,
        $or_intrinsic:ident
    ) => {
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Debug)]
        pub struct $ty(pub ($neon_ty, $neon_ty));

        impl PartialEq for $ty {
            #[inline]
            fn eq(&self, rhs: &$ty) -> bool {
                self.to_array() == rhs.to_array()
            }
        }

        impl Eq for $ty {}

        impl Add for $ty {
            type Output = Self;
            #[inline]
            fn add(self, rhs: $ty) -> Self {
                unsafe {
                    $ty((
                        $add_intrinsic((self.0).0, (rhs.0).0),
                        $add_intrinsic((self.0).1, (rhs.0).1),
                    ))
                }
            }
        }

        impl AddAssign for $ty {
            #[inline]
            fn add_assign(&mut self, rhs: $ty) {
                *self = *self + rhs
            }
        }

        impl Sub for $ty {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: $ty) -> Self {
                unsafe {
                    $ty((
                        $sub_intrinsic((self.0).0, (rhs.0).0),
                        $sub_intrinsic((self.0).1, (rhs.0).1),
                    ))
                }
            }
        }

        impl BitAnd for $ty {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: $ty) -> Self {
                unsafe {
                    $ty((
                        $and_intrinsic((self.0).0, (rhs.0).0),
                        $and_intrinsic((self.0).1, (rhs.0).1),
                    ))
                }
            }
        }

        impl BitXor for $ty {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: $ty) -> Self {
                unsafe {
                    $ty((
                        $xor_intrinsic((self.0).0, (rhs.0).0),
                        $xor_intrinsic((self.0).1, (rhs.0).1),
                    ))
                }
            }
        }

        impl BitOr for $ty {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: $ty) -> Self {
                unsafe {
                    $ty((
                        $or_intrinsic((self.0).0, (rhs.0).0),
                        $or_intrinsic((self.0).1, (rhs.0).1),
                    ))
                }
            }
        }

        impl BitAndAssign for $ty {
            #[inline]
            fn bitand_assign(&mut self, rhs: $ty) {
                *self = *self & rhs;
            }
        }

        impl BitXorAssign for $ty {
            #[inline]
            fn bitxor_assign(&mut self, rhs: $ty) {
                *self = *self ^ rhs;
            }
        }

        impl BitOrAssign for $ty {
            #[inline]
            fn bitor_assign(&mut self, rhs: $ty) {
                *self = *self | rhs;
            }
        }
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// Instantiate types for x86_64
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
impl_shared!(
    u64x4,
    u64,
    _mm256_add_epi64,
    _mm256_sub_epi64,
    _mm256_slli_epi64,
    _mm256_srli_epi64,
    _mm256_extract_epi64
);

#[cfg(target_arch = "x86_64")]
impl_shared!(
    u32x8,
    u32,
    _mm256_add_epi32,
    _mm256_sub_epi32,
    _mm256_slli_epi32,
    _mm256_srli_epi32,
    _mm256_extract_epi32
);

#[cfg(target_arch = "x86_64")]
impl_shared!(
    u32x4,
    u32,
    _mm256_add_epi32,
    _mm256_sub_epi32,
    _mm256_slli_epi32,
    _mm256_srli_epi32,
    _mm256_extract_epi32
);

// ═══════════════════════════════════════════════════════════════════════════
// Instantiate types for aarch64
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
impl_shared!(
    u64x4,
    u64,
    uint64x2_t,
    vaddq_u64,
    vsubq_u64,
    vshlq_n_u64,
    vshrq_n_u64,
    vandq_u64,
    veorq_u64,
    vorrq_u64
);

#[cfg(target_arch = "aarch64")]
impl_shared!(
    u32x8,
    u32,
    uint32x4_t,
    vaddq_u32,
    vsubq_u32,
    vshlq_n_u32,
    vshrq_n_u32,
    vandq_u32,
    veorq_u32,
    vorrq_u32
);

#[cfg(target_arch = "aarch64")]
impl_shared!(
    u32x4,
    u32,
    uint32x4_t,
    vaddq_u32,
    vsubq_u32,
    vshlq_n_u32,
    vshrq_n_u32,
    vandq_u32,
    veorq_u32,
    vorrq_u32
);

// ═══════════════════════════════════════════════════════════════════════════
// Type conversions (x86_64 only for now)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
macro_rules! impl_conv {
    ($src:ident => $($dst:ident),+) => {
        $(
            #[unsafe_target_feature("avx2")]
            impl From<$src> for $dst {
                #[inline]
                fn from(value: $src) -> $dst {
                    $dst(value.0)
                }
            }
        )+
    }
}

#[cfg(target_arch = "x86_64")]
impl_conv!(u64x4 => u32x8);
#[cfg(target_arch = "x86_64")]
impl_conv!(u32x4 => u32x8);

// ═══════════════════════════════════════════════════════════════════════════
// Common type aliases
// ═══════════════════════════════════════════════════════════════════════════

#[allow(non_camel_case_types)]
pub type i64x4 = u64x4;

// Continue with your existing u64x4 implementation...
// (Keep all the rest of your x86_64 code, just gate it properly)

// ═══════════════════════════════════════════════════════════════════════════
// u64x4 implementation - cross-platform
// ═══════════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
impl u64x4 {
    /// A constified variant of `new`.
    /// Should only be called from `const` contexts. At runtime `new` is going to be faster.
    #[inline]
    pub const fn new_const(x0: u64, x1: u64, x2: u64, x3: u64) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // SAFETY: Transmuting between an array and a SIMD type is safe
            // https://rust-lang.github.io/unsafe-code-guidelines/layout/packed-simd-vectors.html
            Self(core::mem::transmute::<[u64; 4], __m256i>([x0, x1, x2, x3]))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u64x4((
                core::mem::transmute([x0, x1]),
                core::mem::transmute([x2, x3]),
            ))
        }
    }

    /// A constified variant of `splat`.
    /// Should only be called from `const` contexts. At runtime `splat` is going to be faster.
    #[inline]
    pub const fn splat_const<const N: u64>() -> Self {
        Self::new_const(N, N, N, N)
    }

    /// Constructs a new instance.
    #[cfg_attr(target_arch = "x86_64", unsafe_target_feature("avx2"))]
    #[inline]
    pub fn new(x0: u64, x1: u64, x2: u64, x3: u64) -> u64x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            u64x4(_mm256_set_epi64x(x3 as i64, x2 as i64, x1 as i64, x0 as i64))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u64x4((
                vld1q_u64([x0, x1].as_ptr()),
                vld1q_u64([x2, x3].as_ptr()),
            ))
        }
    }

    /// Constructs a new instance with all elements initialized to the given value.
    #[cfg_attr(target_arch = "x86_64", unsafe_target_feature("avx2"))]
    #[inline]
    pub fn splat(x: u64) -> u64x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            u64x4(_mm256_set1_epi64x(x as i64))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u64x4((vdupq_n_u64(x), vdupq_n_u64(x)))
        }
    }

    /// Extracts the vector into an array
    #[inline]
    pub fn to_array(self) -> [u64; 4] {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::mem::transmute::<__m256i, [u64; 4]>(self.0)
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let mut out = [0u64; 4];
            vst1q_u64(out.as_mut_ptr(), (self.0).0);
            vst1q_u64(out.as_mut_ptr().add(2), (self.0).1);
            out
        }
    }

    /// Constructs from an array
    #[inline]
    pub fn from_array(arr: [u64; 4]) -> u64x4 {
        u64x4::new(arr[0], arr[1], arr[2], arr[3])
    }

    /// Multiply each lane by a scalar using wrapping multiplication
    #[inline]
    pub fn mul_scalar(self, scalar: u64) -> u64x4 {
        let arr = self.to_array();
        u64x4::new(
            arr[0].wrapping_mul(scalar),
            arr[1].wrapping_mul(scalar),
            arr[2].wrapping_mul(scalar),
            arr[3].wrapping_mul(scalar),
        )
    }

    /// Compare: returns 0xFFFFFFFFFFFFFFFF in lanes where self < rhs, 0 otherwise
    #[inline]
    pub fn cmplt(self, rhs: u64x4) -> u64x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // AVX2 doesn't have unsigned comparison, so use signed comparison trick:
            // XOR with i64::MIN to flip the sign bit
            let flip = u64x4::splat(i64::MIN as u64);
            let a = (self ^ flip).0;
            let b = (rhs ^ flip).0;
            u64x4(_mm256_cmpgt_epi64(b, a))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u64x4((
                vcltq_u64((self.0).0, (rhs.0).0),
                vcltq_u64((self.0).1, (rhs.0).1),
            ))
        }
    }

    /// Alias for cmplt (legacy API compatibility)
    #[inline]
    pub fn cmp_lt(self, rhs: u64x4) -> u64x4 {
        self.cmplt(rhs)
    }

    /// Blend two vectors based on a mask: mask ? true_val : false_val
    /// Lanes where mask is all 1s take from true_val, otherwise from false_val
    #[inline]
    pub fn blend(self, true_val: u64x4, false_val: u64x4) -> u64x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            u64x4(_mm256_blendv_epi8(false_val.0, true_val.0, self.0))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u64x4((
                vbslq_u64((self.0).0, (true_val.0).0, (false_val.0).0),
                vbslq_u64((self.0).1, (true_val.0).1, (false_val.0).1),
            ))
        }
    }

    /// Signed comparison: returns 0xFFFFFFFFFFFFFFFF in lanes where self > rhs (as i64), 0 otherwise
    #[inline]
    pub fn cmp_gt(self, rhs: u64x4) -> u64x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            u64x4(_mm256_cmpgt_epi64(self.0, rhs.0))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u64x4((
                vcgtq_s64(core::mem::transmute((self.0).0), core::mem::transmute((rhs.0).0)),
                vcgtq_s64(core::mem::transmute((self.0).1), core::mem::transmute((rhs.0).1)),
            ))
        }
    }

    /// Constructs from an i64 array (same as from u64 array, two's complement)
    #[inline]
    pub fn from(arr: [i64; 4]) -> u64x4 {
        u64x4::new(arr[0] as u64, arr[1] as u64, arr[2] as u64, arr[3] as u64)
    }
}

// Implement Mul for u64x4 (lane-wise wrapping multiplication)
#[cfg_attr(target_arch = "x86_64", unsafe_target_feature("avx2"))]
impl Mul for u64x4 {
    type Output = u64x4;

    #[inline]
    fn mul(self, rhs: u64x4) -> u64x4 {
        // Both AVX2 and NEON lack efficient 64×64→64 multiply, use scalar
        let a = self.to_array();
        let b = rhs.to_array();
        u64x4::new(
            a[0].wrapping_mul(b[0]),
            a[1].wrapping_mul(b[1]),
            a[2].wrapping_mul(b[2]),
            a[3].wrapping_mul(b[3]),
        )
    }
}

// Implement Shl for u64x4 (runtime shift count using vector shifts)
#[cfg_attr(target_arch = "x86_64", unsafe_target_feature("avx2"))]
impl Shl<u32> for u64x4 {
    type Output = u64x4;

    #[inline]
    fn shl(self, count: u32) -> u64x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let count_vec = u64x4::splat(count as u64);
            u64x4(_mm256_sllv_epi64(self.0, count_vec.0))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u64x4((
                vshlq_u64((self.0).0, vdupq_n_s64(count as i64)),
                vshlq_u64((self.0).1, vdupq_n_s64(count as i64)),
            ))
        }
    }
}

// Implement Shr<u32> for u64x4 (logical right shift with runtime shift count)
#[cfg_attr(target_arch = "x86_64", unsafe_target_feature("avx2"))]
impl Shr<u32> for u64x4 {
    type Output = u64x4;

    #[inline]
    fn shr(self, count: u32) -> u64x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let count_vec = u64x4::splat(count as u64);
            u64x4(_mm256_srlv_epi64(self.0, count_vec.0))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u64x4((
                vshlq_u64((self.0).0, vdupq_n_s64(-(count as i64))),
                vshlq_u64((self.0).1, vdupq_n_s64(-(count as i64))),
            ))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// u32x4 implementation - cross-platform
// ═══════════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
impl u32x4 {
    /// Constructs a new instance
    #[cfg_attr(target_arch = "x86_64", unsafe_target_feature("avx2"))]
    #[inline]
    pub fn new(x0: u32, x1: u32, x2: u32, x3: u32) -> u32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // Use lower 128 bits, upper bits are zero
            u32x4(_mm256_set_epi32(0, 0, 0, 0, x3 as i32, x2 as i32, x1 as i32, x0 as i32))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u32x4((
                vld1q_u32([x0, x1, x2, x3].as_ptr()),
                vdupq_n_u32(0), // padding
            ))
        }
    }

    /// Constructs a new instance with all elements initialized to the given value.
    #[cfg_attr(target_arch = "x86_64", unsafe_target_feature("avx2"))]
    #[inline]
    pub fn splat(x: u32) -> u32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            u32x4(_mm256_set1_epi32(x as i32))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u32x4((vdupq_n_u32(x), vdupq_n_u32(x)))
        }
    }

    /// Constructs from an array
    #[inline]
    pub fn from_array(arr: [u32; 4]) -> u32x4 {
        u32x4::new(arr[0], arr[1], arr[2], arr[3])
    }

    /// Extracts the vector into an array
    #[inline]
    pub fn to_array(self) -> [u32; 4] {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let full: [u32; 8] = core::mem::transmute::<__m256i, [u32; 8]>(self.0);
            [full[0], full[1], full[2], full[3]]
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let mut out = [0u32; 4];
            vst1q_u32(out.as_mut_ptr(), (self.0).0);
            out
        }
    }

    /// Compares for equality
    #[inline]
    pub fn cmpeq(self, rhs: u32x4) -> u32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            u32x4(_mm256_cmpeq_epi32(self.0, rhs.0))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u32x4((
                vceqq_u32((self.0).0, (rhs.0).0),
                vceqq_u32((self.0).1, (rhs.0).1),
            ))
        }
    }

    /// Alias for cmpeq
    #[inline]
    pub fn cmp_eq(self, rhs: u32x4) -> u32x4 {
        self.cmpeq(rhs)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// u32x8 implementation - cross-platform
// ═══════════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
impl u32x8 {
    /// A constified variant of `new`.
    /// Should only be called from `const` contexts. At runtime `new` is going to be faster.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub const fn new_const(
        x0: u32, x1: u32, x2: u32, x3: u32,
        x4: u32, x5: u32, x6: u32, x7: u32,
    ) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // SAFETY: Transmuting between an array and a SIMD type is safe
            // https://rust-lang.github.io/unsafe-code-guidelines/layout/packed-simd-vectors.html
            Self(core::mem::transmute::<[u32; 8], __m256i>([x0, x1, x2, x3, x4, x5, x6, x7]))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u32x8((
                core::mem::transmute([x0, x1, x2, x3]),
                core::mem::transmute([x4, x5, x6, x7]),
            ))
        }
    }

    /// A constified variant of `splat`.
    /// Should only be called from `const` contexts. At runtime `splat` is going to be faster.
    #[inline]
    pub const fn splat_const<const N: u32>() -> Self {
        Self::new_const(N, N, N, N, N, N, N, N)
    }

    /// Constructs a new instance.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(target_arch = "x86_64", unsafe_target_feature("avx2"))]
    #[inline]
    pub fn new(x0: u32, x1: u32, x2: u32, x3: u32, x4: u32, x5: u32, x6: u32, x7: u32) -> u32x8 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            u32x8(_mm256_set_epi32(
                x7 as i32, x6 as i32, x5 as i32, x4 as i32,
                x3 as i32, x2 as i32, x1 as i32, x0 as i32,
            ))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u32x8((
                vld1q_u32([x0, x1, x2, x3].as_ptr()),
                vld1q_u32([x4, x5, x6, x7].as_ptr()),
            ))
        }
    }

    /// Constructs a new instance with all elements initialized to the given value.
    #[cfg_attr(target_arch = "x86_64", unsafe_target_feature("avx2"))]
    #[inline]
    pub fn splat(x: u32) -> u32x8 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            u32x8(_mm256_set1_epi32(x as i32))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u32x8((vdupq_n_u32(x), vdupq_n_u32(x)))
        }
    }

    /// Multiplies the low unsigned 32-bits from each packed 64-bit element
    /// and returns the unsigned 64-bit results.
    /// (This ignores the upper 32-bits from each packed 64-bits!)
    #[inline]
    pub fn mul32(self, rhs: u32x8) -> u64x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm256_mul_epu32(self.0, rhs.0).into()
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            // NEON: vmull_u32 multiplies low 2 lanes to u64x2
            let lo = vmull_u32(vget_low_u32((self.0).0), vget_low_u32((rhs.0).0));
            let hi = vmull_u32(vget_low_u32((self.0).1), vget_low_u32((rhs.0).1));
            u64x4((lo, hi))
        }
    }

    /// Compares for equality, returns 0xFFFFFFFF in lanes where equal, 0 otherwise
    #[inline]
    pub fn cmpeq(self, rhs: u32x8) -> u32x8 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            u32x8(_mm256_cmpeq_epi32(self.0, rhs.0))
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u32x8((
                vceqq_u32((self.0).0, (rhs.0).0),
                vceqq_u32((self.0).1, (rhs.0).1),
            ))
        }
    }

    /// Alias for cmpeq
    #[inline]
    pub fn cmp_eq(self, rhs: u32x8) -> u32x8 {
        self.cmpeq(rhs)
    }

    /// Extracts the vector into an array
    #[inline]
    pub fn to_array(self) -> [u32; 8] {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::mem::transmute::<__m256i, [u32; 8]>(self.0)
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let mut out = [0u32; 8];
            vst1q_u32(out.as_mut_ptr(), (self.0).0);
            vst1q_u32(out.as_mut_ptr().add(4), (self.0).1);
            out
        }
    }

    /// Constructs from an array
    #[inline]
    pub fn from_array(arr: [u32; 8]) -> u32x8 {
        u32x8::new(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7])
    }
}