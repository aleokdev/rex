use ash::vk;
use std::{
    num::NonZeroU64,
    ops::{Bound, RangeBounds},
};

pub fn subresource_range(
    aspect_mask: vk::ImageAspectFlags,
    mip_levels: impl RangeBounds<u32>,
    array_layers: impl RangeBounds<u32>,
) -> vk::ImageSubresourceRange {
    let base_mip_level = match mip_levels.start_bound() {
        Bound::Included(x) => *x,
        Bound::Excluded(x) => *x + 1,
        Bound::Unbounded => 0,
    };

    let level_count = match mip_levels.end_bound() {
        Bound::Included(x) => *x + 1,
        Bound::Excluded(x) => *x,
        Bound::Unbounded => vk::REMAINING_MIP_LEVELS,
    } - base_mip_level;

    let base_array_layer = match array_layers.start_bound() {
        Bound::Included(x) => *x,
        Bound::Excluded(x) => *x + 1,
        Bound::Unbounded => 0,
    };

    let layer_count = match mip_levels.end_bound() {
        Bound::Included(x) => *x + 1,
        Bound::Excluded(x) => *x,
        Bound::Unbounded => vk::REMAINING_ARRAY_LAYERS,
    } - base_array_layer;

    vk::ImageSubresourceRange {
        aspect_mask,
        base_mip_level,
        level_count,
        base_array_layer,
        layer_count,
    }
}

/// Returns `size` as a multiple of `alignment`. Same as `u64::next_multiple_of`.
// TODO: Replace this with `next_multiple_of` once it is stabilized.
// https://github.com/rust-lang/rust/issues/88581
pub fn align(alignment: u64, size: u64) -> u64 {
    (size + alignment - 1) & !(alignment - 1)
}

/// Returns `size` as a multiple of `alignment`. See [align].
pub fn align_nonzero(alignment: NonZeroU64, size: NonZeroU64) -> NonZeroU64 {
    let alignment = alignment.get();
    let size = size.get();

    // TODO: Check if this will ever be zero, and if so, in which circunstances
    NonZeroU64::new((size + alignment - 1) & !(alignment - 1)).unwrap()
}
