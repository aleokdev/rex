use super::{
    buffer::{Buffer, BufferSlice},
    image::Image,
};
use ash::extensions::ext::DebugUtils;
use ash::vk::{self, Handle};
use nonzero_ext::NonZeroAble;
use space_alloc::{
    linear::LinearAllocation, BuddyAllocation, BuddyAllocator, Deallocator, LinearAllocator,
    OutOfMemory,
};
use std::ffi::CStr;
use std::{collections::HashMap, ffi::c_void, num::NonZeroU64};

const DEVICE_BLOCK_SIZE: u64 = 256 * 1024 * 1024;
const HOST_BLOCK_SIZE: u64 = 64 * 1024 * 1024;

#[derive(Debug)]
struct MemoryBlock<Allocator: space_alloc::Allocator> {
    raw: vk::DeviceMemory,
    allocator: Allocator,
    mapped: *mut c_void,
}

impl<Allocator: space_alloc::Allocator> MemoryBlock<Allocator> {
    pub unsafe fn name(
        &self,
        device: vk::Device,
        utils: &DebugUtils,
        name: &CStr,
    ) -> anyhow::Result<()> {
        utils.debug_utils_set_object_name(
            device,
            &vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_handle(self.raw.as_raw())
                .object_name(name)
                .object_type(vk::ObjectType::DEVICE_MEMORY),
        )?;
        Ok(())
    }
}

struct MemoryType<Allocator: space_alloc::Allocator> {
    memory_blocks: Vec<MemoryBlock<Allocator>>,
    memory_props: vk::MemoryPropertyFlags,
    memory_type_index: u32,
    block_size: NonZeroU64,
    min_alloc_size: u64,
    mapped: bool,
}

struct MemoryTypeAllocation<Allocation: space_alloc::Allocation> {
    block_memory: vk::DeviceMemory,
    mapped: *mut c_void,
    block_index: usize,
    allocation: Allocation,
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
        // We iterate through all our blocks until we find one that isn't out of memory, then
        // allocate there.
        for (i, block) in self.memory_blocks.iter_mut().enumerate() {
            match block.allocator.allocate(size, alignment) {
                Ok(allocation) => {
                    use space_alloc::Allocation;
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
        self.allocate_block(device)?;
        self.allocate(device, size, alignment)
    }

    pub unsafe fn destroy(self, device: &ash::Device) {
        self.memory_blocks
            .into_iter()
            .for_each(|block| device.free_memory(block.raw, None));
    }
}

pub struct GpuMemory {
    device: ash::Device,
    memory_props: vk::PhysicalDeviceMemoryProperties,

    linear_linear: HashMap<u32, MemoryType<LinearAllocator>>,
    list_linear: HashMap<u32, MemoryType<BuddyAllocator>>,
    images: HashMap<u32, MemoryType<BuddyAllocator>>,

    scratch: Vec<vk::Buffer>,
}

impl GpuMemory {
    pub unsafe fn new(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<Self> {
        let memory_props = instance.get_physical_device_memory_properties(physical_device);

        Ok(GpuMemory {
            device: device.clone(),
            memory_props,

            linear_linear: HashMap::new(),
            list_linear: HashMap::new(),
            images: HashMap::new(),

            scratch: vec![],
        })
    }

    pub unsafe fn allocate_scratch_buffer(
        &mut self,
        info: vk::BufferCreateInfo,
        usage: MemoryUsage,
        mapped: bool,
    ) -> anyhow::Result<Buffer<LinearAllocation>> {
        let buffer = self.device.create_buffer(&info, None)?;
        let requirements = self.device.get_buffer_memory_requirements(buffer);

        let allocation = match self.allocate_scratch(usage, &requirements, mapped) {
            Ok(x) => x,
            Err(e) => {
                self.device.destroy_buffer(buffer, None);
                return Err(e);
            }
        };

        self.device
            .bind_buffer_memory(buffer, allocation.memory, allocation.offset())?;

        self.scratch.push(buffer);

        Ok(Buffer {
            raw: buffer,
            info,
            allocation,
        })
    }

    pub unsafe fn free_scratch(&mut self) -> anyhow::Result<()> {
        for memory_type in self.linear_linear.values_mut() {
            for block in &mut memory_type.memory_blocks {
                block.allocator.reset();
            }
        }

        for buffer in self.scratch.drain(..) {
            self.device.destroy_buffer(buffer, None);
        }

        Ok(())
    }

    unsafe fn allocate_scratch(
        &mut self,
        usage: MemoryUsage,
        requirements: &vk::MemoryRequirements,
        mapped: bool,
    ) -> anyhow::Result<GpuAllocation<LinearAllocation>> {
        let (memory_type_index, memory_props) = find_properties(
            &self.memory_props,
            requirements.memory_type_bits,
            usage.flags(false),
        )
        .or_else(|| {
            find_properties(
                &self.memory_props,
                requirements.memory_type_bits,
                usage.flags(true),
            )
        })
        .ok_or_else(|| anyhow::anyhow!("no compatible memory on GPU"))?;

        if mapped && !memory_props.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            return Err(anyhow::anyhow!(
                "tried to create mappable memory with non-mappable memory properties"
            ));
        }

        let block_size = if memory_props.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            HOST_BLOCK_SIZE
        } else {
            DEVICE_BLOCK_SIZE
        };

        let memory_type = self
            .linear_linear
            .entry(memory_type_index)
            .or_insert_with(|| MemoryType {
                memory_blocks: vec![],
                memory_props,
                memory_type_index,
                block_size: NonZeroU64::new(block_size).unwrap(),
                min_alloc_size: requirements.alignment,
                mapped,
            });

        let allocation = memory_type.allocate(
            &self.device,
            NonZeroU64::new(requirements.size).unwrap(),
            NonZeroU64::new(requirements.alignment).unwrap(),
        )?;

        Ok(GpuAllocation {
            memory: allocation.block_memory,
            allocation: allocation.allocation,
            memory_type_index,
            memory_block_index: allocation.block_index,
            allocator: AllocatorType::Linear,
            mapped: allocation.mapped,
        })
    }

    pub unsafe fn allocate_buffer(
        &mut self,
        info: vk::BufferCreateInfo,
        usage: MemoryUsage,
        mapped: bool,
    ) -> anyhow::Result<Buffer<BuddyAllocation>> {
        let buffer = self.device.create_buffer(&info, None)?;
        let requirements = self.device.get_buffer_memory_requirements(buffer);

        let allocation = match self.allocate_list(usage, &requirements, mapped) {
            Ok(x) => x,
            Err(e) => {
                self.device.destroy_buffer(buffer, None);
                return Err(e);
            }
        };

        self.device
            .bind_buffer_memory(buffer, allocation.memory, allocation.offset())?;

        Ok(Buffer {
            raw: buffer,
            info,
            allocation,
        })
    }

    pub unsafe fn free_buffer(
        &mut self,
        allocation: GpuAllocation<BuddyAllocation>,
    ) -> anyhow::Result<()> {
        assert_eq!(allocation.allocator, AllocatorType::List);
        self.list_linear
            .get_mut(&allocation.memory_type_index)
            .ok_or_else(|| anyhow::anyhow!("mismatched allocation memory type"))?
            .memory_blocks
            .get_mut(allocation.memory_block_index)
            .ok_or_else(|| anyhow::anyhow!("out of range memory block index"))?
            .allocator
            .deallocate(&allocation.allocation);
        Ok(())
    }

    unsafe fn allocate_list(
        &mut self,
        usage: MemoryUsage,
        requirements: &vk::MemoryRequirements,
        mapped: bool,
    ) -> anyhow::Result<GpuAllocation<BuddyAllocation>> {
        let (memory_type_index, memory_props) = find_properties(
            &self.memory_props,
            requirements.memory_type_bits,
            usage.flags(false),
        )
        .or_else(|| {
            find_properties(
                &self.memory_props,
                requirements.memory_type_bits,
                usage.flags(true),
            )
        })
        .ok_or_else(|| anyhow::anyhow!("no compatible memory on GPU"))?;

        if mapped && !memory_props.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            return Err(anyhow::anyhow!(
                "tried to create mappable memory with non-mappable memory properties"
            ));
        }

        let block_size = if memory_props.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            DEVICE_BLOCK_SIZE
        } else {
            HOST_BLOCK_SIZE
        };

        let memory_type = self
            .list_linear
            .entry(memory_type_index)
            .or_insert_with(|| MemoryType {
                memory_blocks: vec![],
                memory_props,
                memory_type_index,
                block_size: block_size.into_nonzero().unwrap(),
                min_alloc_size: requirements.alignment,
                mapped,
            });

        let allocation = memory_type.allocate(
            &self.device,
            requirements.size.into_nonzero().unwrap(),
            requirements.alignment.into_nonzero().unwrap(),
        )?;

        Ok(GpuAllocation {
            memory: allocation.block_memory,
            allocation: allocation.allocation,
            memory_type_index,
            memory_block_index: allocation.block_index,
            allocator: AllocatorType::List,
            mapped: allocation.mapped,
        })
    }

    pub unsafe fn allocate_image(&mut self, info: &vk::ImageCreateInfo) -> anyhow::Result<Image> {
        let image = self.device.create_image(info, None)?;
        let requirements = self.device.get_image_memory_requirements(image);

        let (memory_type_index, memory_props) = find_properties(
            &self.memory_props,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| anyhow::anyhow!("no compatible memory on GPU"))?;

        let memory_type = self
            .images
            .entry(memory_type_index)
            .or_insert_with(|| MemoryType {
                memory_blocks: vec![],
                memory_props,
                memory_type_index,
                block_size: DEVICE_BLOCK_SIZE.into_nonzero().unwrap(),
                min_alloc_size: requirements.alignment,
                mapped: false,
            });

        let allocation = memory_type.allocate(
            &self.device,
            requirements.size.into_nonzero().unwrap(),
            requirements.alignment.into_nonzero().unwrap(),
        )?;

        use space_alloc::Allocation;
        self.device.bind_image_memory(
            image,
            allocation.block_memory,
            allocation.allocation.offset(),
        )?;

        Ok(Image {
            raw: image,
            allocation: Some(GpuAllocation {
                memory: allocation.block_memory,
                allocation: allocation.allocation,
                memory_type_index,
                memory_block_index: allocation.block_index,
                allocator: AllocatorType::Image,
                mapped: std::ptr::null_mut(),
            }),
            info: *info,
        })
    }

    pub unsafe fn free_image(
        &mut self,
        allocation: GpuAllocation<BuddyAllocation>,
    ) -> anyhow::Result<()> {
        assert_eq!(allocation.allocator, AllocatorType::Image);
        self.images
            .get_mut(&allocation.memory_type_index)
            .ok_or_else(|| anyhow::anyhow!("mismatched allocation memory type"))?
            .memory_blocks
            .get_mut(allocation.memory_block_index)
            .ok_or_else(|| anyhow::anyhow!("out of range memory block index"))?
            .allocator
            .deallocate(&allocation.allocation);
        Ok(())
    }
}

impl Drop for GpuMemory {
    fn drop(&mut self) {
        unsafe { self.free_scratch() };
        self.linear_linear
            .drain()
            .for_each(|(_, mem)| unsafe { mem.destroy(&self.device) });
        self.list_linear
            .drain()
            .for_each(|(_, mem)| unsafe { mem.destroy(&self.device) });
        self.images
            .drain()
            .for_each(|(_, mem)| unsafe { mem.destroy(&self.device) });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AllocatorType {
    Linear,
    List,
    Image,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuAllocation<Allocation: space_alloc::Allocation> {
    pub memory: vk::DeviceMemory,
    pub allocation: Allocation,
    pub memory_type_index: u32,
    pub memory_block_index: usize,
    pub allocator: AllocatorType,
    pub mapped: *mut c_void,
}

impl<Allocation: space_alloc::Allocation> GpuAllocation<Allocation> {
    pub unsafe fn write_mapped<T: Clone>(&self, data: &[T]) -> anyhow::Result<()> {
        if self.mapped.is_null() {
            return Err(anyhow::anyhow!("null mapped ptr"));
        }

        std::slice::from_raw_parts_mut(self.mapped as *mut T, data.len()).clone_from_slice(data);

        Ok(())
    }

    pub fn offset(&self) -> vk::DeviceAddress {
        self.allocation.offset().into()
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.allocation.size().into()
    }

    pub unsafe fn name(
        &self,
        device: vk::Device,
        utils: &DebugUtils,
        name: &CStr,
    ) -> anyhow::Result<()> {
        utils.debug_utils_set_object_name(
            device,
            &vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_handle(self.memory.as_raw())
                .object_name(name)
                .object_type(vk::ObjectType::DEVICE_MEMORY),
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryUsage {
    Gpu,
    CpuToGpu,
    GpuToCpu,
}

impl MemoryUsage {
    fn flags(self, downlevel: bool) -> vk::MemoryPropertyFlags {
        match (self, downlevel) {
            (MemoryUsage::Gpu, _) => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            (MemoryUsage::CpuToGpu, false) => {
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::DEVICE_LOCAL
            }
            (MemoryUsage::GpuToCpu, false) => {
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::HOST_CACHED
            }
            (MemoryUsage::CpuToGpu | MemoryUsage::GpuToCpu, true) => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            }
        }
    }
}

/// Queues a copy from src to dst using a newly allocated scratch buffer.
/// Returns the staging scratch buffer.
pub unsafe fn cmd_stage<T: Clone, Alloc: space_alloc::Allocation>(
    device: &ash::Device,
    scratch: &mut GpuMemory,
    cmd: vk::CommandBuffer,
    src: &[T],
    dst: &BufferSlice<Alloc>,
) -> anyhow::Result<Buffer<LinearAllocation>> {
    let staging = scratch.allocate_scratch_buffer(
        vk::BufferCreateInfo::builder()
            .size(std::mem::size_of::<T>() as u64 * src.len() as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build(),
        MemoryUsage::CpuToGpu,
        true,
    )?;

    staging.allocation.write_mapped(src)?;

    device.cmd_copy_buffer(
        cmd,
        staging.raw,
        dst.buffer.raw,
        &[vk::BufferCopy::builder()
            .src_offset(0)
            .dst_offset(dst.offset)
            .size(staging.info.size)
            .build()],
    );

    Ok(staging)
}

pub unsafe fn cmd_stage_sync(device: &ash::Device, cmd: vk::CommandBuffer) {
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::DependencyFlags::empty(),
        &[vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ)
            .build()],
        &[],
        &[],
    );
}

fn find_properties(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    required_type_bits: u32,
    required_props: vk::MemoryPropertyFlags,
) -> Option<(u32, vk::MemoryPropertyFlags)> {
    for i in 0..memory_props.memory_type_count {
        let type_bits = 1 << i;
        let is_required_type = (required_type_bits & type_bits) != 0;
        let props = memory_props.memory_types[i as usize].property_flags;
        let has_required_props = (props & required_props) == required_props;
        if is_required_type && has_required_props {
            return Some((i, props));
        }
    }
    None
}
