mod mem_type;

pub use self::mem_type::{MemoryBlock, MemoryType, MemoryTypeAllocation};

use super::{
    buffer::{Buffer, BufferSlice},
    image::GpuImage,
};
use ash::extensions::ext::DebugUtils;
use ash::vk::{self, Handle};
use nonzero_ext::{nonzero, NonZeroAble};
use space_alloc::{
    linear::LinearAllocation, BuddyAllocation, BuddyAllocator, Deallocator, LinearAllocator,
};
use std::{collections::HashMap, ffi::c_void, num::NonZeroU64};
use std::{ffi::CStr, sync::Mutex};

/// 256 MiB
const DEVICE_BLOCK_SIZE: NonZeroU64 = nonzero!(256 * 1024 * 1024u64);
/// 64 MiB
const HOST_BLOCK_SIZE: NonZeroU64 = nonzero!(64 * 1024 * 1024u64);
/// 1 KiB
const MIN_ALLOC_SIZE: u64 = 1024u64;

/// Acts as the main allocation point for GPU memory.
///
/// Internally, the memory is divided into [`MemoryType`]s which support specific properties, and those are then divided
/// into [`MemoryBlock`]s which are again divided into [`MemoryTypeAllocation`]s.
pub struct GpuMemory {
    device: ash::Device,
    memory_props: vk::PhysicalDeviceMemoryProperties,

    linear_linear: Mutex<HashMap<u32, MemoryType<LinearAllocator>>>,
    list_linear: Mutex<HashMap<u32, MemoryType<BuddyAllocator>>>,
    images: Mutex<HashMap<u32, MemoryType<BuddyAllocator>>>,

    scratch: Mutex<Vec<vk::Buffer>>,
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

            linear_linear: Default::default(),
            list_linear: Default::default(),
            images: Default::default(),

            scratch: Default::default(),
        })
    }

    pub unsafe fn allocate_scratch_buffer(
        &self,
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

        self.scratch.lock().unwrap().push(buffer);

        Ok(Buffer {
            raw: buffer,
            info,
            allocation,
        })
    }

    unsafe fn allocate_scratch(
        &self,
        usage: MemoryUsage,
        requirements: &vk::MemoryRequirements,
        mapped: bool,
    ) -> anyhow::Result<GpuAllocation<LinearAllocation>> {
        // Find the index & memory properties of a suitable memory type for this allocation.
        let (memory_type_index, memory_props) = find_suitable_memory_type(
            &self.memory_props,
            requirements.memory_type_bits,
            usage.flags(false),
        )
        .or_else(|| {
            find_suitable_memory_type(
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

        // Obtain or create the memory type we are going to allocate into.
        let mut linear_linear = self.linear_linear.lock().unwrap();
        let memory_type = linear_linear
            .entry(memory_type_index)
            .or_insert_with(|| MemoryType {
                memory_blocks: vec![],
                memory_props,
                memory_type_index,
                block_size,
                min_alloc_size: MIN_ALLOC_SIZE,
                mapped,
            });

        // Return an allocation done in the suitable memory type we found.
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
        &self,
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

    unsafe fn allocate_list(
        &self,
        usage: MemoryUsage,
        requirements: &vk::MemoryRequirements,
        mapped: bool,
    ) -> anyhow::Result<GpuAllocation<BuddyAllocation>> {
        let (memory_type_index, memory_props) = find_suitable_memory_type(
            &self.memory_props,
            requirements.memory_type_bits,
            usage.flags(false),
        )
        .or_else(|| {
            find_suitable_memory_type(
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

        let mut list_linear = self.list_linear.lock().unwrap();
        let memory_type = list_linear
            .entry(memory_type_index)
            .or_insert_with(|| MemoryType {
                memory_blocks: vec![],
                memory_props,
                memory_type_index,
                block_size: block_size,
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

    pub unsafe fn allocate_image(&self, info: &vk::ImageCreateInfo) -> anyhow::Result<GpuImage> {
        let image = self.device.create_image(info, None)?;
        let requirements = self.device.get_image_memory_requirements(image);

        let (memory_type_index, memory_props) = find_suitable_memory_type(
            &self.memory_props,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| anyhow::anyhow!("no compatible memory on GPU"))?;

        let mut images = self.images.lock().unwrap();
        let memory_type = images
            .entry(memory_type_index)
            .or_insert_with(|| MemoryType {
                memory_blocks: vec![],
                memory_props,
                memory_type_index,
                block_size: DEVICE_BLOCK_SIZE,
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

        Ok(GpuImage {
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

    pub unsafe fn free_scratch(&self) {
        let mut linear_linear = self.linear_linear.lock().unwrap();
        let mut scratch = self.scratch.lock().unwrap();
        for memory_type in linear_linear.values_mut() {
            for block in &mut memory_type.memory_blocks {
                block.allocator.reset();
            }
        }

        for buffer in scratch.drain(..) {
            self.device.destroy_buffer(buffer, None);
        }
    }

    pub unsafe fn free<Allocation: space_alloc::Allocation>(
        &self,
        allocation: &GpuAllocation<Allocation>,
    ) -> anyhow::Result<()> {
        fn free<Alloc: Deallocator>(
            map: &mut HashMap<u32, MemoryType<Alloc>>,
            allocation: GpuAllocation<Alloc::Allocation>,
        ) -> anyhow::Result<()> {
            map.get_mut(&allocation.memory_type_index)
                .ok_or_else(|| anyhow::anyhow!("mismatched allocation memory type"))?
                .memory_blocks
                .get_mut(allocation.memory_block_index)
                .ok_or_else(|| anyhow::anyhow!("out of range memory block index"))?
                .allocator
                .deallocate(&allocation.allocation);
            Ok(())
        }

        match allocation.allocator {
            AllocatorType::Linear => Ok(()), // We cannot free memory allocated on a linear allocator in a per-alloc basis
            AllocatorType::List => free(
                &mut self.list_linear.lock().unwrap(),
                // HACK because no specialization. this should be safe because allocation should be GpuAllocation<BuddyAllocation>
                std::mem::transmute_copy(allocation),
            ),
            AllocatorType::Image => free(
                &mut self.images.lock().unwrap(),
                // HACK for same reason as above
                std::mem::transmute_copy(allocation),
            ),
        }
    }
}

impl Drop for GpuMemory {
    fn drop(&mut self) {
        unsafe { self.free_scratch() };
        self.linear_linear
            .lock()
            .unwrap()
            .drain()
            .for_each(|(_, mem)| unsafe { mem.destroy(&self.device) });
        self.list_linear
            .lock()
            .unwrap()
            .drain()
            .for_each(|(_, mem)| unsafe { mem.destroy(&self.device) });
        self.images
            .lock()
            .unwrap()
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

/// Searches through a [vk::PhysicalDeviceMemoryProperties] object to find a suitable memory type for the requirements
/// specified through the `required_type_bits` & `required_props` parameters.
///
/// Returns the index of an appropiate memory type to use as well as its property flags.
fn find_suitable_memory_type(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    required_type_bits: u32,
    required_props: vk::MemoryPropertyFlags,
) -> Option<(u32, vk::MemoryPropertyFlags)> {
    (0..memory_props.memory_type_count).find_map(|memory_type_idx| {
        let type_bits = 1 << memory_type_idx;
        let is_required_type = (required_type_bits & type_bits) != 0;
        let props = memory_props.memory_types[memory_type_idx as usize].property_flags;
        let has_required_props = (props & required_props) == required_props;

        (is_required_type && has_required_props).then_some((memory_type_idx, props))
    })
}
