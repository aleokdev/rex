mod block;

use std::{ffi::c_void, num::NonZeroU64};

use ash::vk;
use space_alloc::OutOfMemory;

pub use self::block::MemoryBlock;

pub struct MemoryType<Allocator: space_alloc::Allocator> {
    pub memory_blocks: Vec<MemoryBlock<Allocator>>,
    pub memory_props: vk::MemoryPropertyFlags,
    pub memory_type_index: u32,
    pub block_size: NonZeroU64,
    pub min_alloc_size: u64,
    /// Whether to map the blocks allocated in the memory type into application address space or not.
    pub mapped: bool,
}

pub struct MemoryTypeAllocation<Allocation: space_alloc::Allocation> {
    /// The [vk::DeviceMemory] of the block this allocation was obtained from.
    pub block_memory: vk::DeviceMemory,
    pub mapped: *mut c_void,
    pub block_index: usize,
    pub allocation: Allocation,
}

impl<Allocator: space_alloc::Allocator> MemoryType<Allocator> {
    pub unsafe fn allocate_block(&mut self, device: &ash::Device) -> anyhow::Result<()> {
        let memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::builder()
                .allocation_size(self.block_size.get())
                .memory_type_index(self.memory_type_index as u32),
            None,
        )?;

        let mapped = if self.mapped {
            device.map_memory(
                memory,
                0,
                self.block_size.get(),
                vk::MemoryMapFlags::empty(),
            )?
        } else {
            std::ptr::null_mut()
        };

        self.memory_blocks.push(MemoryBlock {
            raw: memory,
            allocator: Allocator::from_properties(self.min_alloc_size, self.block_size),
            mapped,
        });

        Ok(())
    }

    pub unsafe fn allocate(
        &mut self,
        device: &ash::Device,
        size: NonZeroU64,
        alignment: NonZeroU64,
    ) -> anyhow::Result<MemoryTypeAllocation<Allocator::Allocation>> {
        use space_alloc::Allocation;

        assert!(
            size <= self.block_size,
            "[MemoryType]: tried to allocate space of {}B ({}MiB) which cannot fit in [MemoryBlock]s of size {}B ({}MiB)",
            size,
            size.get() / 1024 / 1024,
            self.block_size,
            self.block_size.get() / 1024 / 1024,
        );

        // We iterate through all our blocks until we find one that isn't out of memory, then
        // allocate there.
        for (i, block) in self.memory_blocks.iter_mut().enumerate() {
            match block.allocator.allocate(size, alignment) {
                Ok(allocation) => {
                    return Ok(MemoryTypeAllocation {
                        block_memory: block.raw,
                        mapped: block.mapped.add(allocation.offset() as usize),
                        allocation,
                        block_index: i,
                    });
                }
                Err(OutOfMemory) => {}
            }
        }
        // If we don't have blocks with free memory left, we allocate a new one and return an allocation from there.
        self.allocate_block(device)?;
        // SAFETY: We have successfully pushed a MemoryBlock in allocate_block. [self.memory_blocks] cannot be empty.
        let block = self.memory_blocks.last_mut().unwrap_unchecked();
        let allocation = block
            .allocator
            .allocate(size, alignment)
            .unwrap_or_else(|_| {
                panic!(
                    "[MemoryType]: could not allocate on newly created memory block: \
            there is most likely a bug in the space_alloc crate \
            (alloc size: {}, alloc alignment: {}, block size: {})",
                    size, alignment, self.block_size
                )
            });

        Ok(MemoryTypeAllocation {
            block_memory: block.raw,
            mapped: block.mapped.add(allocation.offset() as usize),
            allocation,
            block_index: self.memory_blocks.len() - 1,
        })
    }

    pub unsafe fn destroy(self, device: &ash::Device) {
        self.memory_blocks
            .into_iter()
            .for_each(|block| device.free_memory(block.raw, None));
    }
}
