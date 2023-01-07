//! General purpose space allocators.
//! These don't actually allocate any memory, they only provide bookkeeping. Allocation methods will
//! return [Allocation]s with an offset and size that the user can interpret in whichever way they
//! need.
//!
//! In our case, these are used for splitting GPU allocations via [GpuMemory].
//!
//! [GpuMemory]: super::memory::GpuMemory

mod buddy;

use std::num::NonZeroU64;

pub use buddy::{BuddyAllocation, BuddyAllocator};

pub mod linear;

pub use linear::LinearAllocator;

use thiserror::Error;

pub trait Allocation {
    fn offset(&self) -> u64;
    fn size(&self) -> u64;
}

#[derive(Error, Debug)]
#[error("OOM")]
pub struct OutOfMemory;

pub trait Allocator {
    type Allocation: Allocation;

    fn allocate(
        &mut self,
        size: NonZeroU64,
        alignment: NonZeroU64,
    ) -> Result<Self::Allocation, OutOfMemory>;

    fn from_properties(min_alloc: u64, capacity: NonZeroU64) -> Self;
}

pub trait Deallocator: Allocator {
    fn deallocate(&mut self, alloc: &Self::Allocation);
}
