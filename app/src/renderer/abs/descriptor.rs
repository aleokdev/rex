use std::collections::HashMap;

use ash::vk;

use super::Cx;

// credit vblanco

pub struct DescriptorAllocator {
    device: ash::Device,
    current_pool: vk::DescriptorPool,
    free_pools: Vec<vk::DescriptorPool>,
    used_pools: Vec<vk::DescriptorPool>,
}

impl DescriptorAllocator {
    const SIZES: [(vk::DescriptorType, f32); 11] = [
        (vk::DescriptorType::SAMPLER, 0.5),
        (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4.),
        (vk::DescriptorType::SAMPLED_IMAGE, 4.),
        (vk::DescriptorType::STORAGE_IMAGE, 1.),
        (vk::DescriptorType::UNIFORM_TEXEL_BUFFER, 1.),
        (vk::DescriptorType::STORAGE_TEXEL_BUFFER, 1.),
        (vk::DescriptorType::UNIFORM_BUFFER, 2.),
        (vk::DescriptorType::STORAGE_BUFFER, 2.),
        (vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC, 1.),
        (vk::DescriptorType::STORAGE_BUFFER_DYNAMIC, 1.),
        (vk::DescriptorType::INPUT_ATTACHMENT, 0.5),
    ];

    pub fn new(device: &ash::Device) -> Self {
        DescriptorAllocator {
            device: device.clone(),
            current_pool: vk::DescriptorPool::null(),
            free_pools: vec![],
            used_pools: vec![],
        }
    }

    pub unsafe fn reset(&mut self) -> anyhow::Result<()> {
        for pool in &self.used_pools {
            self.device
                .reset_descriptor_pool(*pool, Default::default())?;
        }
        self.free_pools.append(&mut self.used_pools);
        self.current_pool = vk::DescriptorPool::null();
        Ok(())
    }

    pub unsafe fn allocate(
        &mut self,
        layout: vk::DescriptorSetLayout,
    ) -> anyhow::Result<vk::DescriptorSet> {
        if self.current_pool == vk::DescriptorPool::null() {
            self.current_pool = self.acquire_pool()?;
            self.used_pools.push(self.current_pool);
        }

        let mut alloc = vk::DescriptorSetAllocateInfo::builder()
            .set_layouts(&[layout])
            .descriptor_pool(self.current_pool)
            .build();

        match self.device.allocate_descriptor_sets(&alloc) {
            Ok(sets) => return Ok(sets[0]),
            Err(e) => match e {
                vk::Result::ERROR_FRAGMENTED_POOL | vk::Result::ERROR_OUT_OF_POOL_MEMORY => {
                    self.current_pool = self.acquire_pool()?;
                    self.used_pools.push(self.current_pool);
                    alloc.descriptor_pool = self.current_pool;
                    Ok(self.device.allocate_descriptor_sets(&alloc)?[0])
                }
                _ => Err(e.into()),
            },
        }
    }

    unsafe fn acquire_pool(&mut self) -> anyhow::Result<vk::DescriptorPool> {
        if let Some(pool) = self.free_pools.pop() {
            Ok(pool)
        } else {
            Ok(Self::create_pool(&self.device, 1000, Default::default())?)
        }
    }

    unsafe fn create_pool(
        device: &ash::Device,
        n: u32,
        flags: vk::DescriptorPoolCreateFlags,
    ) -> anyhow::Result<vk::DescriptorPool> {
        let sizes = Self::SIZES
            .iter()
            .map(|(ty, size)| {
                vk::DescriptorPoolSize::builder()
                    .ty(*ty)
                    .descriptor_count((size * n as f32) as u32)
                    .build()
            })
            .collect::<Vec<_>>();

        Ok(device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::builder()
                .flags(flags)
                .max_sets(n)
                .pool_sizes(&sizes),
            None,
        )?)
    }
}

impl Drop for DescriptorAllocator {
    fn drop(&mut self) {
        unsafe {
            self.free_pools
                .iter()
                .chain(self.used_pools.iter())
                .for_each(|pool| {
                    self.device.destroy_descriptor_pool(*pool, None);
                });
        }
    }
}

struct DescriptorLayoutInfo {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl std::hash::Hash for DescriptorLayoutInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for binding in &self.bindings {
            binding.binding.hash(state);
            binding.descriptor_type.hash(state);
            binding.descriptor_count.hash(state);
            binding.stage_flags.hash(state);
        }
    }
}

impl PartialEq for DescriptorLayoutInfo {
    fn eq(&self, other: &Self) -> bool {
        if self.bindings.len() != other.bindings.len() {
            false
        } else {
            self.bindings
                .iter()
                .zip(other.bindings.iter())
                .all(|(a, b)| {
                    a.binding == b.binding
                        && a.descriptor_type == b.descriptor_type
                        && a.descriptor_count == b.descriptor_count
                        && a.stage_flags == b.stage_flags
                })
        }
    }
}

impl Eq for DescriptorLayoutInfo {}

pub struct DescriptorLayoutCache {
    device: ash::Device,
    cache: HashMap<DescriptorLayoutInfo, vk::DescriptorSetLayout>,
}

impl DescriptorLayoutCache {
    pub fn new(device: &ash::Device) -> Self {
        DescriptorLayoutCache {
            device: device.clone(),
            cache: HashMap::new(),
        }
    }

    pub unsafe fn create_descriptor_layout(
        &mut self,
        info: &vk::DescriptorSetLayoutCreateInfo,
    ) -> vk::DescriptorSetLayout {
        let mut layout = DescriptorLayoutInfo { bindings: vec![] };
        layout.bindings.reserve(info.binding_count as usize);

        let mut sort = false;
        let mut last = -1i32;

        for binding in std::slice::from_raw_parts(info.p_bindings, info.binding_count as usize) {
            layout.bindings.push(*binding);

            if binding.binding as i32 > last {
                last = binding.binding as i32;
            } else {
                sort = true;
            }
        }

        if sort {
            layout.bindings.sort_by(|a, b| a.binding.cmp(&b.binding));
        }

        *self.cache.entry(layout).or_insert_with(|| {
            self.device
                .create_descriptor_set_layout(info, None)
                .unwrap()
        })
    }
}

impl Drop for DescriptorLayoutCache {
    fn drop(&mut self) {
        self.cache.drain().for_each(|(_, layout)| unsafe {
            self.device.destroy_descriptor_set_layout(layout, None)
        });
    }
}

pub struct DescriptorBuilder {
    writes: Vec<vk::WriteDescriptorSet>,
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl DescriptorBuilder {
    pub fn new() -> Self {
        DescriptorBuilder {
            writes: vec![],
            bindings: vec![],
        }
    }

    pub fn bind_buffer<'a>(
        &'a mut self,
        binding: u32,
        info: &'a [vk::DescriptorBufferInfo],
        ty: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) -> &'a mut Self {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::builder()
                .descriptor_count(1)
                .descriptor_type(ty)
                .stage_flags(stage_flags)
                .binding(binding)
                .build(),
        );

        self.writes.push(
            vk::WriteDescriptorSet::builder()
                .buffer_info(info)
                .descriptor_type(ty)
                .dst_binding(binding)
                .build(),
        );

        self
    }

    pub fn bind_image<'a>(
        &'a mut self,
        binding: u32,
        info: &'a [vk::DescriptorImageInfo],
        ty: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) -> &'a mut Self {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::builder()
                .descriptor_count(1)
                .descriptor_type(ty)
                .stage_flags(stage_flags)
                .binding(binding)
                .build(),
        );

        self.writes.push(
            vk::WriteDescriptorSet::builder()
                .image_info(info)
                .descriptor_type(ty)
                .dst_binding(binding)
                .build(),
        );

        self
    }

    pub unsafe fn build(
        &mut self,
        device: &ash::Device,
        allocator: &mut DescriptorAllocator,
        layout_cache: &mut DescriptorLayoutCache,
    ) -> anyhow::Result<(vk::DescriptorSet, vk::DescriptorSetLayout)> {
        let layout = layout_cache.create_descriptor_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&self.bindings),
        );

        let set = allocator.allocate(layout)?;

        for write in &mut self.writes {
            write.dst_set = set;
        }

        device.update_descriptor_sets(&self.writes, &[]);

        Ok((set, layout))
    }
}
