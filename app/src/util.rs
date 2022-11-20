use ash::vk;
use std::ops::{Bound, RangeBounds};

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
        Bound::Included(x) => *x,
        Bound::Excluded(x) => *x - 1,
        Bound::Unbounded => vk::REMAINING_MIP_LEVELS,
    } - base_mip_level;

    let base_array_layer = match array_layers.start_bound() {
        Bound::Included(x) => *x,
        Bound::Excluded(x) => *x + 1,
        Bound::Unbounded => 0,
    };

    let layer_count = match mip_levels.end_bound() {
        Bound::Included(x) => *x,
        Bound::Excluded(x) => *x - 1,
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
