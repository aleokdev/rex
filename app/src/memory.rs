use crate::{buddy::BuddyAllocator, buffer::Buffer, image::Image};
use ash::vk;
use std::collections::HashMap;
use thiserror::Error;

const DEVICE_BLOCK_SIZE: u64 = 256 * 1024 * 1024;
const HOST_BLOCK_SIZE: u64 = 64 * 1024 * 1024;

#[derive(Error, Debug)]
#[error("OOM")]
struct OutOfMemory;

#[derive(Clone)]
pub enum Allocator {
    Linear { cursor: u64 },
    Buddy(BuddyAllocator),
}

impl Allocator {
    pub unsafe fn allocate(
        &mut self,
        block_size: u64,
        size: u64,
        alignment: u64,
    ) -> anyhow::Result<(u64, u64)> {
        let size = align(alignment, size);
        match self {
            Allocator::Linear { cursor } => {
                if size > block_size {
                    return Err(anyhow::anyhow!("linear allocator: size > block_size"));
                }

                if block_size - *cursor > size {
                    *cursor += size;
                    return Ok((*cursor - size, size));
                }

                Err(anyhow::Error::new(OutOfMemory))
            }
            Allocator::Buddy(allocator) => allocator
                .allocate((u64::BITS - size.leading_zeros()) as u8)
                .ok_or_else(|| anyhow::Error::new(OutOfMemory))
                .map(|offset| (offset, size.next_power_of_two())),
        }
    }

    pub unsafe fn free(&mut self, allocation: Option<&GpuAllocation>) -> anyhow::Result<()> {
        match self {
            Allocator::Linear { cursor } => {
                assert!(allocation.is_none());
                *cursor = 0;
                Ok(())
            }
            Allocator::Buddy(allocator) => allocator.free(),
        }
    }
}

struct MemoryBlock {
    raw: vk::DeviceMemory,
    allocator: Allocator,
}

struct MemoryType {
    memory_blocks: Vec<MemoryBlock>,
    memory_props: vk::MemoryPropertyFlags,
    memory_type_index: u32,
    block_size: u64,
    mappable: bool,
    default_allocator: Allocator,
}

impl MemoryType {
    pub unsafe fn allocate_block(&mut self, device: &ash::Device) -> anyhow::Result<()> {
        let memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::builder()
                .allocation_size(self.block_size)
                .memory_type_index(self.memory_type_index as u32),
            None,
        )?;

        self.memory_blocks.push(MemoryBlock {
            raw: memory,
            allocator: self.default_allocator.clone(),
        });

        Ok(())
    }

    pub unsafe fn allocate(
        &mut self,
        device: &ash::Device,
        size: u64,
        alignment: u64,
    ) -> anyhow::Result<(vk::DeviceMemory, u64, u64)> {
        for block in &mut self.memory_blocks {
            match block.allocator.allocate(self.block_size, size, alignment) {
                Ok((offset, size)) => {
                    return Ok((block.raw, offset, size));
                }
                Err(e) if !e.is::<OutOfMemory>() => {
                    return Err(e);
                }
                _ => {}
            }
        }
        self.allocate_block(device)?;
        self.allocate(device, size, alignment)
    }
}

pub struct GpuMemory {
    device: ash::Device,
    memory_props: vk::PhysicalDeviceMemoryProperties,

    linear_linear: HashMap<u32, MemoryType>,
    list_linear: HashMap<u32, MemoryType>,
    images: HashMap<u32, MemoryType>,
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
        })
    }

    pub unsafe fn allocate_scratch_buffer(
        &mut self,
        info: &vk::BufferCreateInfo,
        usage: MemoryUsage,
        mapped: bool,
    ) -> anyhow::Result<Buffer> {
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
            .bind_buffer_memory(buffer, allocation.memory, allocation.offset)?;

        Ok(Buffer {
            raw: buffer,
            allocation,
        })
    }

    pub unsafe fn free_scratch(&mut self) -> anyhow::Result<()> {
        for memory_type in self.linear_linear.values_mut() {
            for block in &mut memory_type.memory_blocks {
                block.allocator.free(None)?;
            }
        }
        Ok(())
    }

    unsafe fn allocate_scratch(
        &mut self,
        usage: MemoryUsage,
        requirements: &vk::MemoryRequirements,
        mapped: bool,
    ) -> anyhow::Result<GpuAllocation> {
        let (memory_type, memory_props) = find_properties(
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

        let memory_type = self
            .linear_linear
            .entry(memory_type)
            .or_insert_with(|| MemoryType {
                memory_blocks: vec![],
                memory_props,
                memory_type_index: memory_type,
                block_size: if memory_props.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
                    DEVICE_BLOCK_SIZE
                } else {
                    HOST_BLOCK_SIZE
                },
                mappable: memory_props.contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                default_allocator: Allocator::Linear { cursor: 0 },
            });

        let (memory, offset, size) =
            memory_type.allocate(&self.device, requirements.size, requirements.alignment)?;

        Ok(GpuAllocation {
            memory,
            offset,
            size,
        })
    }

    pub unsafe fn allocate_image(&mut self, info: &vk::ImageCreateInfo) -> anyhow::Result<Image> {
        let image = self.device.create_image(info, None)?;
    }
}

#[derive(Debug)]
pub struct GpuAllocation {
    pub memory: vk::DeviceMemory,
    pub offset: vk::DeviceAddress,
    pub size: vk::DeviceSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryUsage {
    Gpu,
    CpuToGpu,
    GpuToCpu,
}

impl MemoryUsage {
    fn flags(self, downlevel: bool) -> vk::MemoryPropertyFlags {
        match self {
            MemoryUsage::Gpu => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryUsage::CpuToGpu if !downlevel => {
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::DEVICE_LOCAL
            }
            MemoryUsage::GpuToCpu if !downlevel => {
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::HOST_CACHED
            }
            MemoryUsage::CpuToGpu | MemoryUsage::GpuToCpu if downlevel => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            }
            _ => unreachable!(),
        }
    }
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

fn align(alignment: u64, size: u64) -> u64 {
    (size + alignment - 1) & !(alignment - 1)
}
