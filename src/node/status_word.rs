use std::borrow::{Borrow, BorrowMut};
use std::fmt::{Debug, Display, Formatter};

#[derive(Clone, PartialEq)]
#[repr(transparent)]
pub struct StatusWord {
    word: u64,
}

impl StatusWord {
    const FROZEN_MASK: u64 = 0x1000_0000_0000_0000;
    const DELETE_SIZE_MASK: u64 = 0x0000_0000_0000_FFFF;
    const RESERVED_RECORD_COUNT_MASK: u64 = 0x0000_0000_FFFF_0000;
    const RESERVED_RECORD_COUNT_SHIFT: u64 = 16;
    const VERSION_MASK: u64 = 0x0000_FFFF_0000_0000;
    const VERSION_SHIFT: u64 = 32;

    const fn zeroed() -> StatusWord {
        StatusWord { word: 0 }
    }

    const fn from(status_word: &StatusWord) -> StatusWordBuilder {
        StatusWordBuilder(StatusWord {
            word: status_word.word,
        })
    }

    pub fn with_records(records: u16) -> StatusWord {
        StatusWordBuilder::new().reserved_records(records).build()
    }

    pub fn froze(&self) -> StatusWord {
        Self::from(self).frozen().build()
    }

    pub fn unfroze(&self) -> StatusWord {
        Self::from(self).unfroze().build()
    }

    pub fn inc_version(&self) -> StatusWord {
        Self::from(self)
            .version(self.version().checked_add(1).unwrap_or(0) as u16)
            .build()
    }

    pub fn reserve_entry(&self) -> StatusWord {
        debug_assert!(
            self.reserved_records() < u16::MAX,
            "Status word can store only {:?} elements",
            u16::MAX
        );
        Self::from(self)
            .reserved_records(self.reserved_records() + 1)
            .version(self.version().checked_add(1).unwrap_or(0) as u16)
            .build()
    }

    pub fn delete_entry(&self) -> StatusWord {
        debug_assert!(
            self.deleted_records() < u16::MAX,
            "Status word can delete only {:?} elements",
            u16::MAX
        );
        Self::from(self)
            .delete_records(self.deleted_records() as u16 + 1)
            .version(self.version().checked_add(1).unwrap_or(0) as u16)
            .build()
    }

    /// Node frozen and cannot be changed
    pub fn is_frozen(&self) -> bool {
        self.word & Self::FROZEN_MASK == Self::FROZEN_MASK
    }

    /// How many records was reserved(but may be not used for KVs) in node
    pub fn reserved_records(&self) -> u16 {
        ((self.word & Self::RESERVED_RECORD_COUNT_MASK) >> Self::RESERVED_RECORD_COUNT_SHIFT) as u16
    }

    /// How many KVs was removed
    pub fn deleted_records(&self) -> u16 {
        (self.word & Self::DELETE_SIZE_MASK) as u16
    }

    /// Return "version" of this node.
    ///
    /// Each change to node increase version. When number of changes overflows `u16` type,
    /// version reset to 0. This field used to detect in-place changes inside interim nodes.
    pub fn version(&self) -> u16 {
        ((self.word & Self::VERSION_MASK) >> Self::VERSION_SHIFT) as u16
    }
}

impl Display for StatusWord {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Frozen: {:?}, reserved records: {:?}, Deleted records: {:?}",
            self.is_frozen(),
            self.reserved_records(),
            self.deleted_records()
        )
    }
}

struct StatusWordBuilder(StatusWord);

impl StatusWordBuilder {
    pub const fn new() -> StatusWordBuilder {
        StatusWordBuilder(StatusWord::zeroed())
    }

    pub fn frozen(&mut self) -> &mut StatusWordBuilder {
        self.0.word |= StatusWord::FROZEN_MASK;
        self
    }

    pub fn unfroze(&mut self) -> &mut StatusWordBuilder {
        self.0.word &= !StatusWord::FROZEN_MASK;
        self
    }

    pub fn reserved_records(&mut self, count: u16) -> &mut StatusWordBuilder {
        self.0.word &= !StatusWord::RESERVED_RECORD_COUNT_MASK;
        self.0.word |= (count as u64) << StatusWord::RESERVED_RECORD_COUNT_SHIFT;
        self
    }

    pub fn version(&mut self, version: u16) -> &mut StatusWordBuilder {
        self.0.word &= !StatusWord::VERSION_MASK;
        self.0.word |= (version as u64) << StatusWord::VERSION_SHIFT;
        self
    }

    pub fn delete_records(&mut self, size: u16) -> &mut StatusWordBuilder {
        self.0.word &= !StatusWord::DELETE_SIZE_MASK;
        self.0.word |= size as u64;
        self
    }

    pub fn build(&mut self) -> StatusWord {
        StatusWord { word: self.0.word }
    }
}

impl Debug for StatusWord {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "status_word: version={};frozen={};reserved={};delete size={}",
            self.version(),
            self.is_frozen(),
            self.reserved_records(),
            self.deleted_records(),
        ))
    }
}

impl Borrow<u64> for StatusWord {
    fn borrow(&self) -> &u64 {
        &self.word
    }
}

impl BorrowMut<u64> for StatusWord {
    fn borrow_mut(&mut self) -> &mut u64 {
        &mut self.word
    }
}

impl From<u64> for StatusWord {
    fn from(word: u64) -> Self {
        StatusWord { word }
    }
}

impl From<StatusWord> for u64 {
    fn from(word: StatusWord) -> u64 {
        word.word
    }
}

#[cfg(test)]
mod tests {
    use crate::node::status_word::{StatusWord, StatusWordBuilder};

    #[test]
    fn create_empty_status_word() {
        let status_word = StatusWord { word: 0 };
        assert!(!status_word.is_frozen());
        assert_eq!(status_word.deleted_records(), 0);
        assert_eq!(status_word.reserved_records(), 0);
    }

    #[test]
    fn create_filled_status_word() {
        let mut status_word = StatusWordBuilder::new().build();
        assert!(!status_word.is_frozen());

        for i in 0u16..u16::MAX {
            status_word = StatusWord::from(&status_word)
                .frozen()
                .delete_records(i as u16)
                .reserved_records(i as u16)
                .build();
            assert!(status_word.is_frozen());
            assert_eq!(status_word.deleted_records(), i);
            assert_eq!(status_word.reserved_records(), i);

            let status_word = status_word.delete_entry();
            assert_eq!(status_word.deleted_records(), i + 1);
            assert!(status_word.is_frozen());
            assert_eq!(status_word.reserved_records(), i);
        }
    }

    #[test]
    fn make_frozen() {
        let mut status_word = StatusWordBuilder::new().build();
        status_word = StatusWord::from(&status_word)
            .delete_records(2)
            .reserved_records(3)
            .build();

        let new_status_word = status_word.froze();
        assert!(!status_word.is_frozen());
        assert_eq!(status_word.deleted_records(), 2);
        assert_eq!(status_word.reserved_records(), 3);

        assert!(new_status_word.is_frozen());
        assert_eq!(new_status_word.deleted_records(), 2);
        assert_eq!(new_status_word.reserved_records(), 3);

        assert!(!new_status_word.unfroze().is_frozen());
        assert_eq!(new_status_word.deleted_records(), 2);
        assert_eq!(new_status_word.reserved_records(), 3);
    }

    #[test]
    fn reserve_entry() {
        let mut status_word = StatusWordBuilder::new().build();
        status_word = StatusWord::from(&status_word)
            .delete_records(2)
            .reserved_records(3)
            .build();

        assert!(!status_word.is_frozen());
        assert_eq!(status_word.deleted_records(), 2);
        assert_eq!(status_word.reserved_records(), 3);

        let new_status_word = status_word.reserve_entry();
        assert!(!new_status_word.is_frozen());
        assert_eq!(new_status_word.deleted_records(), 2);
        assert_eq!(new_status_word.reserved_records(), 4);
    }

    #[test]
    fn version() {
        let mut status_word = StatusWordBuilder::new().build();
        status_word = StatusWord::from(&status_word)
            .version(1)
            .delete_records(2)
            .reserved_records(3)
            .build();

        assert!(!status_word.is_frozen());
        assert_eq!(status_word.version(), 1);
        assert_eq!(status_word.deleted_records(), 2);
        assert_eq!(status_word.reserved_records(), 3);

        status_word = status_word.reserve_entry();
        assert!(!status_word.is_frozen());
        assert_eq!(status_word.version(), 2);
        assert_eq!(status_word.deleted_records(), 2);
        assert_eq!(status_word.reserved_records(), 4);

        status_word = status_word.delete_entry();
        assert!(!status_word.is_frozen());
        assert_eq!(status_word.version(), 3);
        assert_eq!(status_word.deleted_records(), 3);
        assert_eq!(status_word.reserved_records(), 4);

        let mut status_word = StatusWordBuilder::new().build();
        for i in 0..u16::MAX {
            assert!(!status_word.is_frozen());
            assert_eq!(status_word.version(), i);
            status_word = status_word.inc_version();
        }
        status_word = status_word.inc_version();
        assert_eq!(status_word.version(), 0);
    }
}
