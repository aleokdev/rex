use ash::vk;

use crate::abs::{self, descriptor::DescriptorAllocator};

/// Represents an in-flight render frame.
pub struct Frame {
    pub present_semaphore: vk::Semaphore,
    pub render_semaphore: vk::Semaphore,
    pub render_fence: vk::Fence,
    pub cmd_pool: vk::CommandPool,
    pub cmd: vk::CommandBuffer,
    pub allocator: abs::memory::GpuMemory,
    pub deletion: Vec<Box<dyn FnOnce(&mut abs::Cx)>>,
    pub ds_allocator: abs::descriptor::DescriptorAllocator,

    pub counter: u64,
}

impl Frame {
    pub unsafe fn new(cx: &mut abs::Cx) -> anyhow::Result<Self> {
        let render_fence = cx.device.create_fence(
            &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
            None,
        )?;

        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let present_semaphore = cx.device.create_semaphore(&semaphore_info, None)?;
        let render_semaphore = cx.device.create_semaphore(&semaphore_info, None)?;

        let cmd_pool = cx
            .device
            .create_command_pool(&vk::CommandPoolCreateInfo::builder(), None)?;

        // (aleok): We have one fence per frame so only one command buffer is technically required;
        // TODO: We could use multiple so we need to reset less often but I'm leaving that to False
        // P.D: This change is justified since what we were doing previously was allocate one command
        // buffer per frame (and crash because getting out of device memory about 30k frames in)
        let cmd = cx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(1)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(cmd_pool),
        )?[0];

        Ok(Frame {
            present_semaphore,
            render_semaphore,
            render_fence,
            cmd_pool,
            cmd,
            allocator: abs::memory::GpuMemory::new(&cx.device, &cx.instance, cx.physical_device)?,
            deletion: vec![],
            ds_allocator: DescriptorAllocator::new(cx.device.clone()),

            counter: 0,
        })
    }
}
