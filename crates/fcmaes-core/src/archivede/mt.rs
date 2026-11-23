// MT19937 and the two-word canonical binary64 draw used by the supplied libstdc++ driver.
pub(super) struct Mt {
    data: [u32; 624],
    pos: usize,
}
impl Mt {
    pub(super) fn new(seed: u32) -> Self {
        let mut data = [0u32; 624];
        data[0] = seed;
        for i in 1..624 {
            data[i] = (data[i - 1] ^ (data[i - 1] >> 30))
                .wrapping_mul(1812433253)
                .wrapping_add(i as u32);
        }
        Self { data, pos: 624 }
    }
    pub(super) fn next(&mut self) -> u32 {
        if self.pos == 624 {
            #[inline]
            fn word(a: u32, b: u32, c: u32) -> u32 {
                let bits = (a & 0x80000000) | (b & 0x7fffffff);
                c ^ (bits >> 1) ^ ((0u32.wrapping_sub(bits & 1)) & 0x9908b0df)
            }
            for i in 0..227 {
                self.data[i] = word(self.data[i], self.data[i + 1], self.data[i + 397]);
            }
            for i in 227..623 {
                self.data[i] = word(self.data[i], self.data[i + 1], self.data[i - 227]);
            }
            self.data[623] = word(self.data[623], self.data[0], self.data[396]);
            self.pos = 0;
        }
        let mut y = self.data[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^ (y >> 18)
    }
    pub(super) fn unit(&mut self) -> f64 {
        let a = self.next() as f64;
        let b = self.next() as f64;
        ((a + b * 4294967296.) / 18446744073709551616.).min(f64::from_bits(1f64.to_bits() - 1))
    }
}
