// https://github.com/Restioson/buddy-allocator-workshop

#[derive(Debug, Clone)]
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
            for block in tree.iter_mut().skip(start).take(size) {
                *block = Block::new_free(order);
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
    fn block(&self, i: usize) -> &Block {
        &self.tree[i]
    }

    #[inline]
    fn block_mut(&mut self, i: usize) -> &mut Block {
        &mut self.tree[i]
    }

    pub fn allocate(&mut self, desired_order: u8) -> Option<u64> {
        assert!(desired_order <= self.max_order);

        let root = self.block_mut(0);

        if root.order_free == 0 || (root.order_free - 1) < desired_order {
            return None;
        }

        let mut addr: u64 = 0;
        let mut node_index = 1;

        let max_level = self.max_order - desired_order;

        for level in 0..max_level {
            let left_child_index = node_index << 1;
            let left_child = self.block(left_child_index - 1);

            let o = left_child.order_free;
            node_index = if o != 0 && o > desired_order {
                left_child_index
            } else {
                addr += 1 << ((self.max_order_size - level - 1) as u64);
                left_child_index + 1
            };
        }

        let block = self.block_mut(node_index - 1);
        block.order_free = 0;

        for _ in 0..max_level {
            let right_index = node_index & !1;
            node_index >>= 1;

            let left = self.block(right_index - 1).order_free;
            let right = self.block(right_index).order_free;

            self.block_mut(node_index - 1).order_free = std::cmp::max(left, right);
        }

        Some(addr)
    }

    pub fn deallocate(&mut self, offset: u64, order: u8) {
        assert!(order <= self.max_order);

        let level = self.max_order - order;
        let level_offset = Self::blocks_in_tree(level);
        let index = level_offset + ((offset as usize) >> (order + self.base_order)) + 1;

        assert!(index < Self::blocks_in_tree(self.level_count));
        assert_eq!(self.block(index - 1).order_free, 0);

        self.block_mut(index - 1).order_free = order + 1;

        self.update_blocks_above(index, order);
    }

    fn update_block(&mut self, node_index: usize, order: u8) {
        assert!(order != 0);
        assert!(node_index != 0);

        let left_index = (node_index << 1) - 1;
        let left = self.block(left_index).order_free;
        let right = self.block(left_index + 1).order_free;

        if left == order && right == order {
            self.block_mut(node_index - 1).order_free = order + 1;
        } else {
            self.block_mut(node_index - 1).order_free = std::cmp::max(left, right);
        }
    }

    fn update_blocks_above(&mut self, index: usize, order: u8) {
        let mut node_index = index;

        for order in order + 1..=self.max_order {
            node_index >>= 1;
            self.update_block(node_index, order);
        }
    }

    const fn blocks_in_tree(levels: u8) -> usize {
        ((1 << levels) - 1) as _
    }
}

pub fn order_of(x: u64, base_order: u8) -> u8 {
    if x == 0 {
        return 0;
    }

    let mut i = x;
    let mut log2 = 0;
    while i > 0 {
        i >>= 1;
        log2 += 1;
    }

    let log2 = log2;

    if log2 > base_order {
        log2 - base_order
    } else {
        0
    }
}

#[derive(Debug, Clone)]
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
