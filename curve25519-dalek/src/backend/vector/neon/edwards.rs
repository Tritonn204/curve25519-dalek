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

//! Parallel Edwards Arithmetic for Curve25519 on ARM NEON.
//!
//! This module implements vectorized Edwards arithmetic using ARM NEON intrinsics.
//! Unlike the AVX2 backend which packs 4 field elements, NEON packs 2 field elements
//! due to its 128-bit register width.
//!
//! This module has two point types:
//!
//! * `ExtendedPoint`: a point stored in vector-friendly format, with
//!   vectorized doubling and addition (packing 2 coordinates per operation);
//!
//! * `CachedPoint`: used for readdition.
//!
//! The formulas are similar to the AVX2 backend but adapted for 2-wide SIMD.

#![allow(non_snake_case)]

use core::ops::{Add, Neg, Sub};

use subtle::Choice;
use subtle::ConditionallySelectable;

use crate::edwards;
use crate::window::{LookupTable, NafLookupTable5};

#[cfg(any(feature = "precomputed-tables", feature = "alloc"))]
use crate::window::NafLookupTable8;

use crate::traits::Identity;

use super::constants;
use super::field::{FieldElement51x2, Lanes};

/// A point on Curve25519, using parallel Edwards formulas for curve
/// operations on ARM NEON.
///
/// This packs 2 field elements at a time (X,Y or Z,T pairs) for SIMD operations.
///
/// # Invariant
///
/// The coefficients of an `ExtendedPoint` are kept reduced.
#[derive(Copy, Clone, Debug)]
pub struct ExtendedPoint {
    /// Stores (X, Y) in the first vector
    pub(super) xy: FieldElement51x2,
    /// Stores (Z, T) in the second vector
    pub(super) zt: FieldElement51x2,
}

impl From<edwards::EdwardsPoint> for ExtendedPoint {
    fn from(P: edwards::EdwardsPoint) -> ExtendedPoint {
        ExtendedPoint {
            xy: FieldElement51x2::new(&P.X, &P.Y),
            zt: FieldElement51x2::new(&P.Z, &P.T),
        }
    }
}

impl From<ExtendedPoint> for edwards::EdwardsPoint {
    fn from(P: ExtendedPoint) -> edwards::EdwardsPoint {
        let xy = P.xy.split();
        let zt = P.zt.split();
        edwards::EdwardsPoint {
            X: xy[0],
            Y: xy[1],
            Z: zt[0],
            T: zt[1],
        }
    }
}

impl ConditionallySelectable for ExtendedPoint {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        ExtendedPoint {
            xy: FieldElement51x2::conditional_select(&a.xy, &b.xy, choice),
            zt: FieldElement51x2::conditional_select(&a.zt, &b.zt, choice),
        }
    }

    fn conditional_assign(&mut self, other: &Self, choice: Choice) {
        self.xy.conditional_assign(&other.xy, choice);
        self.zt.conditional_assign(&other.zt, choice);
    }
}

impl Default for ExtendedPoint {
    fn default() -> ExtendedPoint {
        ExtendedPoint::identity()
    }
}

impl Identity for ExtendedPoint {
    fn identity() -> ExtendedPoint {
        constants::EXTENDEDPOINT_IDENTITY
    }
}

impl ExtendedPoint {
    /// Compute the double of this point.
    pub fn double(&self) -> ExtendedPoint {
        let xy_split = self.xy.split();
        let zt_split = self.zt.split();
        
        // X + Y
        let x_plus_y = &xy_split[0] + &xy_split[1];
        
        // Square: X^2, Y^2, Z^2
        let x_sq = xy_split[0].square();
        let y_sq = xy_split[1].square();
        let z_sq = zt_split[0].square();
        
        // (X+Y)^2
        let x_plus_y_sq = x_plus_y.square();
        
        // S1 = X^2 + Y^2
        // S2 = X^2 - Y^2 (note: this is negative of what we want for standard formula)
        // S3 = 2*Z^2
        let s1 = &x_sq + &y_sq;
        let s2 = &x_sq - &y_sq;
        let s3 = &z_sq + &z_sq;  // 2*Z^2 via addition
        
        // S8 = S3 + S2 = 2*Z^2 + X^2 - Y^2
        // S9 = S1 - (X+Y)^2 = X^2 + Y^2 - X^2 - 2XY - Y^2 = -2XY
        let s8 = &s3 + &s2;
        let s9 = &s1 - &x_plus_y_sq;
        
        // Result: (X3, Y3, Z3, T3) = (S8*S9, S1*S2, S8*S2, S1*S9)
        ExtendedPoint {
            xy: FieldElement51x2::new(&(&s8 * &s9), &(&s1 * &s2)),
            zt: FieldElement51x2::new(&(&s8 * &s2), &(&s1 * &s9)),
        }
    }

    pub fn mul_by_pow_2(&self, k: u32) -> ExtendedPoint {
        let mut tmp: ExtendedPoint = *self;
        for _ in 0..k {
            tmp = tmp.double();
        }
        tmp
    }
}

/// A cached point with some precomputed variables used for readdition.
///
/// # Warning
///
/// It is not safe to negate this point more than once.
#[derive(Copy, Clone, Debug)]
pub struct CachedPoint {
    /// Stores (Y-X, Y+X) in the first vector
    pub(super) yx: FieldElement51x2,
    /// Stores (2Z, 2dT) in the second vector  
    pub(super) z2t2: FieldElement51x2,
}

impl From<ExtendedPoint> for CachedPoint {
    fn from(P: ExtendedPoint) -> CachedPoint {
        let xy_split = P.xy.split();
        let zt_split = P.zt.split();
        
        let y_minus_x = &xy_split[1] - &xy_split[0];
        let y_plus_x = &xy_split[1] + &xy_split[0];
        
        let z2 = &zt_split[0] + &zt_split[0];
        let t2d = &zt_split[1] * &constants::EDWARDS_D2;
        
        CachedPoint {
            yx: FieldElement51x2::new(&y_minus_x, &y_plus_x),
            z2t2: FieldElement51x2::new(&z2, &t2d),
        }
    }
}

impl Default for CachedPoint {
    fn default() -> CachedPoint {
        CachedPoint::identity()
    }
}

impl Identity for CachedPoint {
    fn identity() -> CachedPoint {
        constants::CACHEDPOINT_IDENTITY
    }
}

impl ConditionallySelectable for CachedPoint {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        CachedPoint {
            yx: FieldElement51x2::conditional_select(&a.yx, &b.yx, choice),
            z2t2: FieldElement51x2::conditional_select(&a.z2t2, &b.z2t2, choice),
        }
    }

    fn conditional_assign(&mut self, other: &Self, choice: Choice) {
        self.yx.conditional_assign(&other.yx, choice);
        self.z2t2.conditional_assign(&other.z2t2, choice);
    }
}

impl Neg for &CachedPoint {
    type Output = CachedPoint;
    
    /// Lazily negate the point by swapping Y-X and Y+X, and negating 2dT.
    ///
    /// # Warning
    ///
    /// Because this method does not perform a reduction, it is not
    /// safe to repeatedly negate a point.
    fn neg(self) -> CachedPoint {
        let yx_split = self.yx.split();
        let z2t2_split = self.z2t2.split();
        
        // Swap Y-X and Y+X (effectively negating X)
        let yx_negated = FieldElement51x2::new(&yx_split[1], &yx_split[0]);
        // Negate 2dT
        let z2t2_negated = FieldElement51x2::new(&z2t2_split[0], &(-&z2t2_split[1]));
        
        CachedPoint {
            yx: yx_negated,
            z2t2: z2t2_negated,
        }
    }
}

impl Add<&CachedPoint> for &ExtendedPoint {
    type Output = ExtendedPoint;

    /// Add an `ExtendedPoint` and a `CachedPoint`.
    fn add(self, other: &CachedPoint) -> ExtendedPoint {
        let xy_split = self.xy.split();
        let zt_split = self.zt.split();
        let yx_split = other.yx.split();
        let z2t2_split = other.z2t2.split();
        
        // Unified addition formula
        let y_minus_x = &xy_split[1] - &xy_split[0];
        let y_plus_x = &xy_split[1] + &xy_split[0];
        
        // A = (Y1-X1)*(Y2-X2)
        // B = (Y1+X1)*(Y2+X2)
        let a = &y_minus_x * &yx_split[0];
        let b = &y_plus_x * &yx_split[1];
        
        // C = 2d*T1*T2
        // D = 2*Z1*Z2
        let c = &zt_split[1] * &z2t2_split[1];
        let d = &zt_split[0] * &z2t2_split[0];
        
        // E = B - A
        // F = D - C
        // G = D + C
        // H = B + A
        let e = &b - &a;
        let f = &d - &c;
        let g = &d + &c;
        let h = &b + &a;
        
        // X3 = E*F
        // Y3 = G*H
        // Z3 = F*G
        // T3 = E*H
        ExtendedPoint {
            xy: FieldElement51x2::new(&(&e * &f), &(&g * &h)),
            zt: FieldElement51x2::new(&(&f * &g), &(&e * &h)),
        }
    }
}

impl Sub<&CachedPoint> for &ExtendedPoint {
    type Output = ExtendedPoint;

    /// Implement subtraction by negating the point and adding.
    fn sub(self, other: &CachedPoint) -> ExtendedPoint {
        self + &(-other)
    }
}

impl From<&edwards::EdwardsPoint> for LookupTable<CachedPoint> {
    fn from(point: &edwards::EdwardsPoint) -> Self {
        let P = ExtendedPoint::from(*point);
        let mut points = [CachedPoint::from(P); 8];
        for i in 0..7 {
            points[i + 1] = (&P + &points[i]).into();
        }
        LookupTable(points)
    }
}

impl From<&edwards::EdwardsPoint> for NafLookupTable5<CachedPoint> {
    fn from(point: &edwards::EdwardsPoint) -> Self {
        let A = ExtendedPoint::from(*point);
        let mut Ai = [CachedPoint::from(A); 8];
        let A2 = A.double();
        for i in 0..7 {
            Ai[i + 1] = (&A2 + &Ai[i]).into();
        }
        // Now Ai = [A, 3A, 5A, 7A, 9A, 11A, 13A, 15A]
        NafLookupTable5(Ai)
    }
}

#[cfg(any(feature = "precomputed-tables", feature = "alloc"))]
impl From<&edwards::EdwardsPoint> for NafLookupTable8<CachedPoint> {
    fn from(point: &edwards::EdwardsPoint) -> Self {
        let A = ExtendedPoint::from(*point);
        let mut Ai = [CachedPoint::from(A); 64];
        let A2 = A.double();
        for i in 0..63 {
            Ai[i + 1] = (&A2 + &Ai[i]).into();
        }
        // Now Ai = [A, 3A, 5A, 7A, 9A, 11A, 13A, 15A, ..., 127A]
        NafLookupTable8(Ai)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn addition_test_helper(P: edwards::EdwardsPoint, Q: edwards::EdwardsPoint) {
        let cached_Q = CachedPoint::from(ExtendedPoint::from(Q));
        let R_vector: edwards::EdwardsPoint = (&ExtendedPoint::from(P) + &cached_Q).into();
        let S_vector: edwards::EdwardsPoint = (&ExtendedPoint::from(P) - &cached_Q).into();

        assert_eq!(R_vector.compress(), (&P + &Q).compress());
        assert_eq!(S_vector.compress(), (&P - &Q).compress());
    }

    #[test]
    fn vector_addition_vs_edwards_extendedpoint() {
        use crate::constants;
        use crate::scalar::Scalar;

        let P = edwards::EdwardsPoint::identity();
        let Q = edwards::EdwardsPoint::identity();
        addition_test_helper(P, Q);

        let P = edwards::EdwardsPoint::identity();
        let Q = constants::ED25519_BASEPOINT_POINT;
        addition_test_helper(P, Q);

        let P = constants::ED25519_BASEPOINT_POINT;
        let Q = constants::ED25519_BASEPOINT_POINT;
        addition_test_helper(P, Q);

        let P = constants::ED25519_BASEPOINT_POINT;
        let Q = constants::ED25519_BASEPOINT_TABLE * &Scalar::from(8475983829u64);
        addition_test_helper(P, Q);
    }

    fn doubling_test_helper(P: edwards::EdwardsPoint) {
        let R: edwards::EdwardsPoint = ExtendedPoint::from(P).double().into();
        assert_eq!(R.compress(), (&P + &P).compress());
    }

    #[test]
    fn vector_doubling_vs_edwards_extendedpoint() {
        use crate::constants;
        use crate::scalar::Scalar;

        let P = edwards::EdwardsPoint::identity();
        doubling_test_helper(P);

        let P = constants::ED25519_BASEPOINT_POINT;
        doubling_test_helper(P);

        let P = constants::ED25519_BASEPOINT_TABLE * &Scalar::from(8475983829u64);
        doubling_test_helper(P);
    }
}