//! Table generation module. This module is public but should be treated as
//! unstable. The API may change without notice.

pub mod generation;

use bytemuck::{Pod, Zeroable};
use std::{
    mem::size_of,
    ops::ControlFlow,
};
use crate::field::FieldElement;

use super::affine_montgomery::AffineMontgomeryPoint;

pub(crate) const L2: usize = 9; // corresponds to a batch size of 256 and a T2 table of a few Ko.
pub(crate) const BATCH_SIZE: usize = 1 << (L2 - 1);

pub(crate) const I_BITS: usize = L2 - 1;
pub(crate) const I_MAX: usize = (1 << I_BITS) + 1; // there needs to be one more element in T2
pub(crate) const CUCKOO_K: usize = 3; // number of cuckoo lookups before giving up

// Note: file layout is just T2 followed by T1 keys and then T1 values.
// We just do casts using `bytemuck` since everything are PODs.

/// A view into an ECDLP precomputed table. This is a wrapper around a read-only byte array, which you could back by an memory mapped file, for example.
pub struct ECDLPTablesFileView<'a> {
    bytes: &'a [u8],
    l1: usize,
}

impl<'a> ECDLPTablesFileView<'a> {
    /// Calculate the cuckoo length for a given `l1` const.
    pub fn cuckoo_len(l1: usize) -> usize {
        let j_max: u64 = 1 << (l1 - 1);
        // x1.3 is the load factor of the cuckoo hashmap.
        ((j_max * 30 / 100) + j_max) as usize
    }

    /// ECDLP algorithm may panic if the alignment or size of `bytes` is wrong.
    pub fn from_bytes(bytes: &'a [u8], l1: usize) -> Self {
        // TODO(merge): check align/size of `bytes` here
        Self { bytes, l1 }
    }

    /// Get the T1 table.
    pub(crate) fn get_t1(&self) -> CuckooT1HashMapView<'_> {
        let t1_keys_values: &[u32] =
            bytemuck::cast_slice(&self.bytes[(size_of::<T2MontgomeryCoordinates>() * I_MAX)..]);

        let cuckoo_len = Self::cuckoo_len(self.l1);
        CuckooT1HashMapView {
            keys: &t1_keys_values[0..cuckoo_len],
            values: &t1_keys_values[cuckoo_len..],
            cuckoo_len,
        }
    }

    /// Get the `l1` constant used to generate the tables.
    #[inline(always)]
    pub fn get_l1(&self) -> usize {
        self.l1
    }

    /// Get the T2 table.
    pub(crate) fn get_t2(&self) -> T2LinearTableView<'_> {
        let t2: &[T2MontgomeryCoordinates] =
            bytemuck::cast_slice(&self.bytes[0..(size_of::<T2MontgomeryCoordinates>() * I_MAX)]);
        T2LinearTableView(t2)
    }
}

/// Canonical FieldElement type.
type CompressedFieldElement = [u8; 32];

#[repr(C, align(32))]
#[derive(Clone, Copy, Default, Pod, Zeroable, Debug)]
/// An entry in the T2 table. Represents a (u,v) coordinate on the Montgomery curve.
pub(crate) struct T2MontgomeryCoordinates {
    /// The `u` coordinate.
    pub u: CompressedFieldElement,
    /// The `v` coordinate.
    pub v: CompressedFieldElement,
}

impl From<T2MontgomeryCoordinates> for AffineMontgomeryPoint {
    fn from(e: T2MontgomeryCoordinates) -> Self {
        Self {
            u: FieldElement::from_bytes(&e.u),
            v: FieldElement::from_bytes(&e.v),
        }
    }
}

impl From<AffineMontgomeryPoint> for T2MontgomeryCoordinates {
    fn from(e: AffineMontgomeryPoint) -> Self {
        Self {
            u: e.u.to_bytes(),
            v: e.v.to_bytes(),
        }
    }
}

/// A view into the T2 table.
pub(crate) struct T2LinearTableView<'a>(pub &'a [T2MontgomeryCoordinates]);

impl T2LinearTableView<'_> {
    #[inline]
    pub fn index(&self, index: usize) -> AffineMontgomeryPoint {
        let T2MontgomeryCoordinates { u, v } = self.0[index];
        AffineMontgomeryPoint::from_bytes(&u, &v)
    }
}

/// A view into the T1 table.
pub(crate) struct CuckooT1HashMapView<'a> {
    /// Cuckoo keys
    pub keys: &'a [u32],
    /// Cuckoo values
    pub values: &'a [u32],
    /// Cuckoo length for provided L1
    pub cuckoo_len: usize,
}

impl CuckooT1HashMapView<'_> {
    pub(crate) fn lookup(
        &self,
        x: &[u8],
        mut is_problem_answer: impl FnMut(u64) -> bool,
    ) -> Option<u64> {
        for i in 0..CUCKOO_K {
            let start = i * 8;
            let end = start + 4;
            let key = u32::from_be_bytes(x[end..end + 4].try_into().expect("key u32"));
            let h = u32::from_be_bytes(x[start..start + 4].try_into().expect("h u32")) as usize
                % self.cuckoo_len;
            if self.keys[h] == key {
                let value = self.values[h] as u64;
                if is_problem_answer(value) {
                    return Some(value);
                }
            }
        }
        None
    }
}

/// A progress report step.
/// This is used to report progress to the user and give additional context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportStep {
    /// Generating T1 points.
    T1PointsGeneration,
    /// Setting up the T1 cuckoo hashmap.
    T1CuckooSetup,
    /// Generating the T2 table.
    T2Table,
}

/// A trait for reporting progress during table generation.
pub trait ProgressTableGenerationReportFunction {
    /// Run the progress report function.
    fn report(&self, progress: f64, step: ReportStep) -> ControlFlow<()>;
}

/// A no-op progress report function.
pub struct NoOpProgressTableGenerationReportFunction;

impl ProgressTableGenerationReportFunction for NoOpProgressTableGenerationReportFunction {
    fn report(&self, _progress: f64, _step: ReportStep) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

/// A trait for automatically converting a closure into a progress report function.
impl<F: Fn(f64, ReportStep) -> ControlFlow<()>> ProgressTableGenerationReportFunction for F {
    #[inline(always)]
    fn report(&self, progress: f64, step: ReportStep) -> ControlFlow<()> {
        self(progress, step)
    }
}