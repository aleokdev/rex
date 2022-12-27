//! General purpose space allocators.
//! These don't actually allocate any memory, they only provide bookkeeping. Allocation methods will
//! return [Allocation]s with an offset and size that the user can interpret in whichever way they
//! need.
//!
//! In our case, these are used for splitting GPU allocations via [GpuMemory].
//!
//! [GpuMemory]: super::memory::GpuMemory

mod buddy;

pub use buddy::BuddyAllocator;

pub mod linear;

pub use linear::LinearAllocator;

use thiserror::Error;

pub struct Allocation {
    pub offset: u64,
    pub size: u64,
}

#[derive(Error, Debug)]
#[error("OOM")]
pub struct OutOfMemory;
