use mwcas::U64Pointer;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(transparent)]
pub struct Metadata {
    word: u64,
}

impl Metadata {
    const NOT_USED_MASK: u64 = 0x0000_0000_0000_0004;
    const RESERVED_MASK: u64 = 0x0000_0000_0000_0002;
    const DELETED_MASK: u64 = 0x0000_0000_0000_0001;
    const VISIBLE_MASK: u64 = 0x0000_0000_0000_0000;

    #[inline(always)]
    pub fn visible() -> Metadata {
        Metadata {
            word: Self::VISIBLE_MASK,
        }
    }

    #[inline(always)]
    pub fn reserved() -> Metadata {
        Metadata {
            word: Self::RESERVED_MASK,
        }
    }

    #[inline(always)]
    pub fn deleted() -> Metadata {
        Metadata {
            word: Self::DELETED_MASK,
        }
    }

    #[inline(always)]
    pub fn not_used() -> Metadata {
        Metadata {
            word: Self::NOT_USED_MASK,
        }
    }

    #[inline(always)]
    pub fn visible_or_deleted(&self) -> bool {
        self.word < Self::RESERVED_MASK
    }

    #[inline(always)]
    pub fn is_visible(&self) -> bool {
        self.word == Self::VISIBLE_MASK
    }

    #[inline(always)]
    pub fn is_deleted(&self) -> bool {
        self.word == Self::DELETED_MASK
    }

    #[inline(always)]
    pub fn is_reserved(&self) -> bool {
        self.word == Self::RESERVED_MASK
    }
}

impl From<u64> for Metadata {
    fn from(word: u64) -> Self {
        Metadata { word }
    }
}

impl From<Metadata> for u64 {
    fn from(word: Metadata) -> Self {
        word.word
    }
}

impl From<Metadata> for U64Pointer {
    fn from(word: Metadata) -> Self {
        Self::new(word.word)
    }
}

impl From<U64Pointer> for Metadata {
    fn from(word: U64Pointer) -> Self {
        Self {
            word: word.read(&crossbeam_epoch::pin()),
        }
    }
}
