use crate::abs::util::align;

/// Allocation information from a `BuddyAllocator` allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BuddyAllocation {
    offset: u64,
    size: u64,
    index: usize,
}

impl BuddyAllocation {
    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}

/// Bitmap-tree-based [buddy allocator], loosely based on https://github.com/Restioson/buddy-allocator-workshop.
///
/// [buddy allocator]: https://en.wikipedia.org/wiki/Buddy_memory_allocation
#[derive(Clone)]
pub struct BuddyAllocator {
    blocks: Vec<Block>,
    o0_size: u64,
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
                string.push_str(format!("{:x}", self.read_block(index)).as_str());
                string.extend(std::iter::repeat(' ').take(item_spacing));
                index += 1;
            }
            string.push('\n');
        }

        f.write_str(&string)
    }
}

impl BuddyAllocator {
    /// Creates a new buddy allocator.
    ///
    /// - `max_order` specifies the max allocatable order.
    /// - `o0_size` specifies the physical size of a block of order 0.
    pub fn new(max_order: u8, o0_size: u64) -> Self {
        let mut max_level = max_order + 1;
        let mut blocks = vec![Block::new(0); Self::blocks_for_level(max_level)];
        let mut i = 0;
        for o in 0..max_level {
            let n = 1 << o;
            blocks[i..(i + n)].fill(Block::new(max_order - o));
            i += n;
        }
        BuddyAllocator {
            blocks,
            o0_size,
            max_order,
        }
    }

    /// Allocates memory of physical size `size`.
    pub fn allocate(&mut self, size: u64) -> Option<BuddyAllocation> {
        let aligned = align(self.o0_size, size);
        let order = Self::order_of(aligned / self.o0_size);

        let root = self.read_block(0);
        if root == 0 || root - 1 < order {
            // root == 0: root is allocated, i.e., there is 0 memory left.
            // root - 1 < order: greatest available order cannot satisfy req allocation.
            return None;
        }
        // past this point we know that there is enough memory for req allocation.

        // fast path: allocating all of it
        if root - 1 == order {
            self.write_block(0, 0);
            return Some(BuddyAllocation {
                offset: 0,
                size: aligned,
                index: 0,
            });
        }

        // start from the top.
        // keep going down until we hit a block that matches order.
        let max_level = self.max_order - order;
        let mut offset = 0; // physical offset into blocks
        let mut i = 0; // current index
        for level in 0..max_level {
            let i_parent = i;
            let parent = self.read_block(i_parent);
            let i_left = Self::left_child(i);
            let left = self.read_block(i_left);

            // check if left can satsify req allocation
            if left != 0 && left - 1 >= order {
                i = i_left;
            } else {
                // otherwise, we know for certain that the right must then be able to
                // because the parent's order said it could (which is the max of left/right)
                i = i_left + 1;
                offset += 1 << ((self.max_order - level - 1) as u64);
            }
        }

        self.write_block(i, 0);
        self.update(i, max_level);
        Some(BuddyAllocation {
            offset: self.o0_size * offset,
            size: aligned,
            index: i,
        })
    }

    /// Deallocates some previously allocated memory.
    pub fn deallocate(&mut self, alloc: BuddyAllocation) {
        // deallocation routine is very simple because we have
        // the luxury of storing index with the allocation
        let order = Self::order_of(alloc.size / self.o0_size);
        self.write_block(alloc.index, order + 1);
        self.update(alloc.index, self.max_order - order);
    }

    fn read_block(&self, i: usize) -> u8 {
        self.blocks[i].order
    }

    fn write_block(&mut self, i: usize, new_order: u8) {
        self.blocks[i].order = new_order;
    }

    fn update(&mut self, mut i: usize, max_level: u8) {
        // traverse upwards and set parent order to max of child order
        for _ in 0..max_level {
            // ensure we start from right child (we don't know if i is left or right)
            let i_right = (i + 1) & !1;
            i = Self::parent(i);
            let left = self.read_block(i_right - 1);
            let right = self.read_block(i_right);
            self.write_block(i, left.max(right)); // parent = max children
        }
    }

    const fn blocks_for_level(level: u8) -> usize {
        ((1 << level) - 1) as usize
    }

    const fn left_child(i: usize) -> usize {
        ((i + 1) << 1) - 1
    }

    const fn right_child(i: usize) -> usize {
        (((i + 1) << 1) + 1) - 1
    }

    const fn parent(i: usize) -> usize {
        ((i + 1) >> 1) - 1
    }

    pub const fn order_of(x: u64) -> u8 {
        if x == 0 {
            return 0;
        }
        let mut i = x;
        let mut log2 = 0;
        while i > 0 {
            i >>= 1;
            log2 += 1;
        }
        log2
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Block {
    // >0: greatest order - 1 available to allocate from this subtree
    //      (including from this block itself)
    // =0: allocated
    order: u8,
}

impl Block {
    pub fn new(order: u8) -> Self {
        Block { order: order + 1 }
    }
}

#[cfg(test)]
mod test {
    use super::BuddyAllocator;

    #[test]
    fn empty() {
        let allocator = BuddyAllocator::new(4, 1);
        println!("{:?}", allocator);
    }

    #[test]
    fn allocate() {
        //    1
        //  0   1
        // 0 0 0 1
        let mut allocator = BuddyAllocator::new(3, 1);
        allocator.allocate(0);
        allocator.allocate(0);
        allocator.allocate(0);
        println!("{:?}", allocator);
    }
}
