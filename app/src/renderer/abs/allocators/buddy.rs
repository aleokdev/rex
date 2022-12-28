use std::num::NonZeroU64;

use crate::{abs::util::align, renderer::abs::util::align_nonzero};

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
    pub fn allocate(&mut self, size: NonZeroU64) -> Option<BuddyAllocation> {
        let aligned = if let Some(o0_size) = NonZeroU64::new(self.o0_size) {
            align_nonzero(o0_size, size)
        } else {
            size
        };
        let order = Self::order_of(size, self.o0_size);

        let root = self.read_block(0);
        if root == 0 || root - 1 < order {
            // root == 0: root is allocated, i.e., there is 0 memory left.
            // root - 1 < order: greatest available order cannot satisfy req allocation.
            return None;
        }
        // past this point we know that there is enough memory for req allocation.

        // start from the top.
        // keep going down until we hit a block that matches order.
        let max_level = self.max_order - order;
        let mut offset = 0; // physical offset into blocks
        let mut i = 0; // current index
        for level in 0..max_level {
            let i_parent = i;
            let i_left = Self::left_child(i_parent);
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
            size: aligned.get(),
            index: i,
        })
    }

    /// Deallocates some previously allocated memory.
    pub fn deallocate(&mut self, alloc: BuddyAllocation) {
        // deallocation routine is very simple because we have
        // the luxury of storing index with the allocation
        let order = Self::order_of(NonZeroU64::new(alloc.size).unwrap(), self.o0_size); // HACK: Use NonZeroU64 in alloc data instead
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

    pub fn order_of(size: NonZeroU64, o0_size: u64) -> u8 {
        // SAFETY: ceil div of any number that is not zero will always be bigger than 0.
        let o0_blocks_needed = unsafe { NonZeroU64::new_unchecked(div_ceil(size.get(), o0_size)) };
        log2_ceil(o0_blocks_needed) as u8
    }
}

fn div_ceil(a: u64, b: u64) -> u64 {
    (a + b - 1) / b
}

pub const fn log2_ceil(x: NonZeroU64) -> u32 {
    u64::BITS - (x.get() - 1).leading_zeros()
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
    use nonzero_ext::nonzero;

    #[test]
    fn empty() {
        let allocator = BuddyAllocator::new(4, 1);
        println!("{:?}", allocator);
        assert_eq!(
            allocator.blocks,
            vec![
                5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1
            ]
            .into_iter()
            .map(|order| super::Block { order })
            .collect::<Vec<_>>()
        )
    }

    #[test]
    fn order() {
        assert_eq!(BuddyAllocator::order_of(nonzero!(1u64), 1), 0);
        assert_eq!(
            BuddyAllocator::order_of(nonzero!(100u64), 1),
            // Needs at least 100 o0 blocks -> 128 -> 2^7
            7
        );
        assert_eq!(
            BuddyAllocator::order_of(nonzero!(50u64), 10),
            // Needs at least 5 o0 blocks -> 8 -> 2^3
            3
        );
    }

    #[test]
    fn allocate_single() {
        //    2
        //  1   2
        // 0 1 1 1
        let mut allocator = BuddyAllocator::new(2, 1);
        allocator.allocate(nonzero!(1u64));
        println!("{:?}", allocator);
        assert_eq!(
            allocator.blocks,
            vec![2, 1, 2, 0, 1, 1, 1]
                .into_iter()
                .map(|order| super::Block { order })
                .collect::<Vec<_>>()
        )
    }

    #[test]
    fn allocate_small() {
        //    1
        //  0   1
        // 0 0 0 1
        let mut allocator = BuddyAllocator::new(2, 1);
        allocator.allocate(nonzero!(1u64));
        allocator.allocate(nonzero!(1u64));
        allocator.allocate(nonzero!(1u64));
        println!("{:?}", allocator);
        assert_eq!(
            allocator.blocks,
            vec![1, 0, 1, 0, 0, 0, 1]
                .into_iter()
                .map(|order| super::Block { order })
                .collect::<Vec<_>>()
        )
    }

    #[test]
    fn allocate_mixed() {
        //    1
        //  1   0
        // 0 1 1 1
        let mut allocator = BuddyAllocator::new(2, 1);
        allocator.allocate(nonzero!(1u64));
        allocator.allocate(nonzero!(2u64));
        println!("{:?}", allocator);
        assert_eq!(
            allocator.blocks,
            vec![1, 1, 0, 0, 1, 1, 1]
                .into_iter()
                .map(|order| super::Block { order })
                .collect::<Vec<_>>()
        )
    }

    // TODO: Test deallocation
}
