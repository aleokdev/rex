// https://github.com/Restioson/buddy-allocator-workshop

#[derive(Clone)]
pub struct BuddyAllocator {
    tree: Vec<Block>,

    level_count: u8,
    max_order: u8,
    base_order: u8,
    max_order_size: u8,
}

impl BuddyAllocator {
    pub fn new(level_count: u8, base_order: u8) -> Self {
        let blocks_in_tree = BuddyAllocator::blocks_in_tree(level_count);
        let max_order = level_count - 1;
        let max_order_size = base_order + max_order;

        let mut tree = vec![Block::new_free(0); blocks_in_tree];

        let mut start = 0usize;
        for level in 0..level_count {
            let order = max_order - level;
            let size = 1 << (level as usize);
            for block in start..(start + size) {
                tree[block] = Block::new_free(order);
            }
            start += size;
        }

        BuddyAllocator {
            tree,

            level_count,
            max_order,
            base_order,
            max_order_size,
        }
    }

    #[inline]
    unsafe fn block(&self, i: usize) -> &Block {
        self.tree.get_unchecked(i)
    }

    #[inline]
    unsafe fn block_mut(&mut self, i: usize) -> &mut Block {
        self.tree.get_unchecked_mut(i)
    }

    pub fn allocate(&mut self, desired_order: u8) -> Option<u64> {
        let root = unsafe { self.block_mut(0) };

        if root.order_free == 0 || (root.order_free - 1) < desired_order {
            return None;
        }

        let mut addr: u64 = 0;
        let mut node_index = 1;

        let max_level = self.max_order - desired_order;

        for level in 0..max_level {
            let left_child_index = node_index << 1;
            let left_child = unsafe { self.block(left_child_index - 1) };

            let o = left_child.order_free;
            node_index = if o != 0 && o > desired_order {
                left_child_index
            } else {
                addr += 1 << ((self.max_order_size - level - 1) as u64);
                left_child_index + 1
            };
        }

        let block = unsafe { self.block_mut(node_index - 1) };
        block.order_free = 0;

        for _ in 0..max_level {
            let right_index = node_index & !1;
            node_index = node_index >> 1;

            let left = unsafe { self.block(right_index - 1) }.order_free;
            let right = unsafe { self.block(right_index) }.order_free;

            unsafe { self.block_mut(node_index - 1) }.order_free = std::cmp::max(left, right);
        }

        Some(addr)
    }

    const fn blocks_in_tree(levels: u8) -> usize {
        ((1 << levels) - 1) as _
    }
}

#[derive(Clone)]
struct Block {
    order_free: u8,
}

impl Block {
    pub fn new_free(order: u8) -> Self {
        Block {
            order_free: order + 1,
        }
    }
}
