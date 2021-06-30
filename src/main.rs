extern crate  packed_simd_2;
extern crate rayon_cond;

use std::arch::x86_64::{
    __m128i, _mm_extract_epi8, _mm_load_si128, _mm_setr_epi8, _mm_shuffle_epi8,
};
use rayon_cond::CondIterator;
use std::sync::atomic::{AtomicI32, Ordering};

const MAX_N: usize = 16;
const MAX_BLOCKS: usize = 24;

fn compute_factorials(n: usize) -> [u32; MAX_N] {
    let mut factorials = [1; 16];
    for i in 1..n + 1 {
        factorials[i] = factorials[i - 1] * i as u32;
    }
    factorials
}


pub fn get_blocks_and_size(perm_max: usize) -> (usize, usize) {
    if perm_max < MAX_BLOCKS {
        (1, perm_max)
    } else {
        (
            MAX_BLOCKS + if perm_max % MAX_BLOCKS == 0 { 0 } else { 1 },
            perm_max / MAX_BLOCKS,
        )
    }
}


pub fn create_count_current(
    n: usize,
    start: usize,
    factorials: &[u32; MAX_N],
) -> (__m128i, [u32; MAX_N]) {
    let mut count = [0u32; MAX_N];
    let mut start = start as u32;
    let mut temp = [0i8; 16];
    let mut current_aux = [0i8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    for i in (1..n).rev() {
        let d = start / factorials[i];
        start %= factorials[i];
        count[i] = d;
        temp.copy_from_slice(&current_aux);
        let d = d as usize;
        for j in 0..i + 1 {
            current_aux[j] = if j + d <= i {
                temp[j + d]
            } else {
                temp[j + d - i - 1]
            };
        }
    }

    unsafe { (_mm_load_si128(std::mem::transmute(&current_aux)), count) }
}

#[inline(always)]
pub fn increment_permutation(count: &mut [u32; MAX_N], mask_shift:&[__m128i;16], mut current: __m128i) -> __m128i {
    unsafe {
        let mut i = 1;
        loop {
            current = _mm_shuffle_epi8(current, mask_shift[i]);
            count[i] += 1;
            if count[i] <= i as u32 {
                break;
            }
            count[i] = 0;
            i += 1;
        }
        current
    }
}

#[inline(always)]
fn reverse(x: __m128i, idx: usize,maks_reverse:&[__m128i;16]) -> __m128i {
    unsafe { _mm_shuffle_epi8(x, maks_reverse[idx]) }
}


static CHECKSUM:AtomicI32=AtomicI32::new(0);
static MAX_FLIPS:AtomicI32=AtomicI32::new(0);

fn main() {
    let n = std::env::args()
        .nth(1)
        .and_then(|n| n.parse().ok())
        .unwrap_or(7);

    let masks_reverse: [__m128i; 16] = unsafe {
        [
            _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            _mm_setr_epi8(1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(4, 3, 2, 1, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(6, 5, 4, 3, 2, 1, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 12, 13, 14, 15),
            _mm_setr_epi8(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 12, 13, 14, 15),
            _mm_setr_epi8(12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14, 15),
            _mm_setr_epi8(13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 14, 15),
            _mm_setr_epi8(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15),
            _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        ]
    };

    let masks_shift: [__m128i; 16] = unsafe {
        [
            _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            _mm_setr_epi8(1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 11, 12, 13, 14, 15),
            _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 15),
            _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 12, 13, 14, 15),
            _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 13, 14, 15),
            _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14, 15),
            _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15),
            _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0),
        ]
    };

    let factorials = compute_factorials(n);
    let perm_max = factorials[n as usize] as usize;
    let (blocks, block_size) = get_blocks_and_size(perm_max);

    CondIterator::new(0..blocks,true).for_each(|i_block|{
        let block_start = i_block * block_size;
        let masks_reverse = masks_reverse.clone();
        let masks_shift = masks_shift.clone();
        let (mut current, mut count) = create_count_current(n, block_start, &factorials);
        let mut current_start = current;
        let mut checksum =0;
        let mut max_flips=0;
        unsafe {
            let mut first = _mm_extract_epi8::<0>(current) as usize;
            let mut crt_idx = block_start;
            let block_end = block_start + block_size;
            while crt_idx < block_end {
                if first > 0 {
                    let mut flips = 0;
                    let mut next = std::mem::transmute::<__m128i, [u8; 16]>(current)[first];
                    while next != 0 {
                        current = reverse(current, first,&masks_reverse);
                        first = next as usize;
                        next = std::mem::transmute::<__m128i, [u8; 16]>(current)[first];
                        flips += 1;
                    }

                    checksum += if crt_idx % 2 == 0 { flips } else { -flips };
                    if flips > max_flips {
                        max_flips = flips;
                    }
                }
                current = increment_permutation(&mut count, &masks_shift,current_start);
                current_start = current;
                first = _mm_extract_epi8::<0>(current) as usize;
                crt_idx += 1;
            }

            CHECKSUM.fetch_add(checksum,Ordering::Release);
            let check_flips=MAX_FLIPS.load(Ordering::Acquire);
            if max_flips>check_flips {
                MAX_FLIPS.store(max_flips, Ordering::Release);
            }
        }
    });


    println!("{}", CHECKSUM.load(Ordering::Acquire));
    println!("Pfannkuchen({})={}", n, MAX_FLIPS.load(Ordering::Acquire));
}