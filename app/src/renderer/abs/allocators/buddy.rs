/// A [buddy allocator], adapted from https://github.com/Restioson/buddy-allocator-workshop.
///
/// [buddy allocator]: https://en.wikipedia.org/wiki/Buddy_memory_allocation
#[derive(Clone)]
pub struct BuddyAllocator {
    tree: Vec<Block>,

    level_count: u8,
    max_order: u8,
}

impl std::fmt::Debug for BuddyAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut string = String::new();
        let mut index = 0;
        for level in 0..=self.max_order {
            let order = self.max_order - level;
            let start_spacing = 2usize.pow(order as u32) - 1;
            let item_spacing = 2 * start_spacing + 1;
            string.extend(std::iter::repeat(' ').take(start_spacing));
            for _block in 0..2usize.pow(level as u32) {
                string.push(('0' as u8 + self.block(index).order_free) as char);
                string.extend(std::iter::repeat(' ').take(item_spacing));
                index += 1;
            }
            string.push('\n');
        }

        f.write_str(&string)
    }
}

impl BuddyAllocator {
    pub fn new(level_count: u8) -> Self {
        let blocks_in_tree = Self::blocks_in_tree(level_count);
        let max_order = level_count - 1;

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
        }
    }

    /// Returns a reference to a block given its identifier (index).
    #[inline]
    fn block(&self, i: usize) -> &Block {
        &self.tree[i]
    }

    /// Returns a mutable reference to a block given its identifier (index).
    #[inline]
    fn block_mut(&mut self, i: usize) -> &mut Block {
        &mut self.tree[i]
    }

    /// Allocates a block of the order given. Returns the index of the resulting block.
    ///
    /// If no blocks of the desired order are available, the function will return None.
    pub fn allocate(&mut self, desired_order: u8) -> Option<u64> {
        assert!(
            desired_order <= self.max_order,
            "invalid allocation order: greater than allocator max order"
        );

        let root = self.block_mut(0);

        // If the root node has no orders free, or if it does not have the desired order free, we
        // return no allocation.
        if root.order_free == 0 || (root.order_free - 1) < desired_order {
            return None;
        }

        let mut addr: u64 = 0;
        // To simplify the operations done we operate with the actual node index + 1. So since the
        // root is 0, we start at 0 + 1 = 1.
        let mut node_index = 1;

        let max_level = self.max_order - desired_order;

        for level in 0..max_level {
            let left_child_index = node_index << 1;
            let left_child = self.block(left_child_index - 1);

            let o = left_child.order_free;
            // If the child is not occupied (o!=0) and (desired_order in o-1)
            // Due to the +1 offset, we need to subtract 1 from 0:
            // However, (o - 1) >= desired_order can be simplified to o > desired_order.
            node_index = if o != 0 && o > desired_order {
                left_child_index
            } else {
                // Move over to the right: if the parent had a free order and the left didn't, the
                // right must, or the parent is invalid and does not uphold invariants
                // Since the address is moving from the left hand side, we need to increase it
                // Block size in bytes = 2^(BASE_ORDER + order)
                // We also only want to allocate on the order of the child, hence subtracting 1
                addr += 1 << ((self.max_order - level - 1) as u64);
                left_child_index + 1
            };
        }

        // Take the block allocated and set its free orders to 0, since the entire block has been
        // used and there is no more space to give out.
        let block = self.block_mut(node_index - 1);
        block.order_free = 0;

        // Iterate upwards and set parents accordingly.
        for _ in 0..max_level {
            let right_index = node_index & !1;

            let left = self.block(right_index - 1).order_free;
            let right = self.block(right_index).order_free;

            node_index >>= 1;
            let parent = self.block_mut(node_index - 1);
            parent.order_free = std::cmp::max(left, right);
        }

        Some(addr)
    }

    pub fn deallocate(&mut self, offset: u64, order: u8) {
        // REFACTOR What is offset?
        assert!(order <= self.max_order);

        let level = self.max_order - order;
        let level_offset = Self::blocks_in_tree(level);
        let index = level_offset + ((offset as usize) >> (order)) + 1;

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
        // REFACTOR why isn't x.log2() being used here instead?

        if log2 > base_order {
            log2 - base_order
        } else {
            0
        }
    }
}

#[derive(Debug, Clone)]
struct Block {
    /// The maximum order allocation that can be allocated with this block and its children.
    ///
    /// For instance, for a fully free tree the order_free value of all blocks looks like this:
    /// ```text
    ///        3
    ///    2       2
    ///  1   1   1   1
    /// 0 0 0 0 0 0 0 0
    /// ```
    ///
    /// For a tree with one allocated order 0 block marked with a T:
    /// ```text
    ///        2
    ///    1       2
    ///  0   1   1   1
    /// T 0 0 0 0 0 0 0
    /// ```
    order_free: u8,
}

impl Block {
    pub fn new_free(order: u8) -> Self {
        Block {
            order_free: order + 1,
        }
    }
}

#[cfg(test)]
mod test {
    use super::BuddyAllocator;

    #[test]
    fn empty() {
        let allocator = BuddyAllocator::new(4);
        println!("{:?}", allocator);
    }

    #[test]
    fn allocate() {
        //    0
        //  0   0
        // T T T 0
        let mut allocator = BuddyAllocator::new(3);
        allocator.allocate(0);
        allocator.allocate(0);
        allocator.allocate(0);
        println!("{:?}", allocator);
    }
}
