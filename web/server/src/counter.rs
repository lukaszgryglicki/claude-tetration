//! How often the two programs were asked for a calculation, and by how
//! many different clients.
//!
//! The counts are kept in a small file, when one is named, so that a
//! restart does not begin at zero: the file is written again a few seconds
//! after a count has changed. A client is told from another by the address
//! that its request came from; of the addresses only hashes are kept, made
//! with a secret of the file, never the addresses themselves. An IPv6
//! address counts with its first 64 bits, the network of one client.

use std::collections::HashMap;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

/// The clients that are told apart; beyond them a new one is not counted.
const CLIENTS: usize = 200_000;

/// The first line of the file.
const HEAD: &str = "tet-web counter 1";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Program {
    Tet,
    Sm,
}

impl Program {
    fn bit(self) -> u8 {
        match self {
            Program::Tet => 1,
            Program::Sm => 2,
        }
    }
}

/// The counts of a moment.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Counts {
    /// When the counting began, in seconds of the epoch.
    pub since: u64,
    pub tet: u64,
    pub sm: u64,
    /// The clients that asked either program, `tet`, `symbolic-math`.
    pub clients: u64,
    pub tet_clients: u64,
    pub sm_clients: u64,
    /// There are more clients than are told apart.
    pub full: bool,
}

#[derive(Default)]
struct State {
    since: u64,
    secret: (u64, u64),
    tet: u64,
    sm: u64,
    /// The hash of a client with the programs that it asked.
    clients: HashMap<u64, u8>,
    /// A client was turned away because the table is full.
    full: bool,
    /// Something changed since the file was written.
    changed: bool,
}

pub struct Counter {
    state: Mutex<State>,
    file: Option<PathBuf>,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// Two numbers that nobody can guess, from what the standard library
/// offers: the keys that it draws for its hash tables.
fn secret() -> (u64, u64) {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let draw = |salt: u64| {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(salt ^ u64::from(std::process::id()));
        hasher.write_u128(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |since| since.as_nanos()),
        );
        hasher.finish()
    };
    (draw(1), draw(2))
}

/// What tells a client from another: the four bytes of an IPv4 address,
/// the first eight of an IPv6 address, or the text when it is no address.
fn identity(client: &str) -> Vec<u8> {
    match client
        .trim()
        .parse::<IpAddr>()
        .map(|address| address.to_canonical())
    {
        Ok(IpAddr::V4(address)) => address.octets().to_vec(),
        Ok(IpAddr::V6(address)) => address.octets()[..8].to_vec(),
        Err(_) => client.trim().as_bytes().to_vec(),
    }
}

/// FNV-1a over the secret and the bytes: the same on every system and
/// with every compiler, as a file that outlives the program needs it.
fn hashed(secret: (u64, u64), bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let keyed = secret
        .0
        .to_le_bytes()
        .into_iter()
        .chain(secret.1.to_le_bytes())
        .chain(bytes.iter().copied())
        .chain(secret.1.to_le_bytes());
    for byte in keyed {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // (the last bytes reach every bit)
    hash ^= hash >> 29;
    hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash ^ (hash >> 32)
}

impl State {
    fn fresh() -> State {
        State {
            since: now(),
            secret: secret(),
            ..State::default()
        }
    }

    /// The state that a file holds; an error says what is wrong with it.
    fn read(text: &str) -> Result<State, String> {
        let mut lines = text.lines();
        if lines.next() != Some(HEAD) {
            return Err(format!("it does not begin with {HEAD:?}"));
        }
        let mut state = State::default();
        let number = |text: &str| {
            text.parse::<u64>()
                .map_err(|_| format!("{text:?} is no number"))
        };
        let hex = |text: &str| {
            u64::from_str_radix(text, 16).map_err(|_| format!("{text:?} is no hexadecimal number"))
        };
        for line in lines.filter(|line| !line.trim().is_empty()) {
            let words: Vec<&str> = line.split_whitespace().collect();
            match words.as_slice() {
                ["since", value] => state.since = number(value)?,
                ["secret", first, second] => state.secret = (hex(first)?, hex(second)?),
                ["tet", value] => state.tet = number(value)?,
                ["sm", value] => state.sm = number(value)?,
                ["full"] => state.full = true,
                ["client", hash, programs] => {
                    let programs = number(programs)?;
                    if !(1..=3).contains(&programs) {
                        return Err(format!("a client asked the programs {programs}"));
                    }
                    state.clients.insert(hex(hash)?, programs as u8);
                }
                _ => return Err(format!("the line {line:?} is not understood")),
            }
        }
        if state.since == 0 || state.secret == (0, 0) {
            return Err("it has no beginning or no secret".into());
        }
        Ok(state)
    }

    fn written(&self) -> String {
        let mut text = format!(
            "{HEAD}\nsince {}\nsecret {:016x} {:016x}\ntet {}\nsm {}\n",
            self.since, self.secret.0, self.secret.1, self.tet, self.sm
        );
        if self.full {
            text.push_str("full\n");
        }
        let mut clients: Vec<(&u64, &u8)> = self.clients.iter().collect();
        clients.sort();
        for (hash, programs) in clients {
            text.push_str(&format!("client {hash:016x} {programs}\n"));
        }
        text
    }

    fn counts(&self) -> Counts {
        let asked = |program: Program| {
            self.clients
                .values()
                .filter(|programs| *programs & program.bit() != 0)
                .count() as u64
        };
        Counts {
            since: self.since,
            tet: self.tet,
            sm: self.sm,
            clients: self.clients.len() as u64,
            tet_clients: asked(Program::Tet),
            sm_clients: asked(Program::Sm),
            full: self.full,
        }
    }
}

impl Counter {
    /// The counter of `file`, which is read when it is there. A file that
    /// cannot be understood is set aside (as `file.unread`) and the
    /// counting begins anew; the text says so.
    pub fn open(file: Option<PathBuf>) -> (Counter, Option<String>) {
        let mut remark = None;
        let state = match file.as_deref().map(std::fs::read_to_string) {
            Some(Ok(text)) => State::read(&text).unwrap_or_else(|reason| {
                let file = file.as_deref().unwrap_or(Path::new(""));
                let aside = file.with_extension("unread");
                let kept = match std::fs::rename(file, &aside) {
                    Ok(()) => format!("it is kept as {}", aside.display()),
                    Err(error) => format!("it could not be set aside ({error})"),
                };
                remark = Some(format!(
                    "the counter file {} is not understood ({reason}): the counting begins anew, {kept}",
                    file.display()
                ));
                State::fresh()
            }),
            Some(Err(error)) if error.kind() != std::io::ErrorKind::NotFound => {
                remark = Some(format!(
                    "the counter file {} cannot be read ({error}): the counting begins anew",
                    file.as_deref().unwrap_or(Path::new("")).display()
                ));
                State::fresh()
            }
            _ => State::fresh(),
        };
        let counter = Counter {
            state: Mutex::new(state),
            file,
        };
        (counter, remark)
    }

    /// One more request for a calculation of `program`, from `client`.
    pub fn count(&self, program: Program, client: &str) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        match program {
            Program::Tet => state.tet += 1,
            Program::Sm => state.sm += 1,
        }
        let hash = hashed(state.secret, &identity(client));
        if state.clients.len() < CLIENTS || state.clients.contains_key(&hash) {
            *state.clients.entry(hash).or_insert(0) |= program.bit();
        } else {
            state.full = true;
        }
        state.changed = true;
    }

    pub fn counts(&self) -> Counts {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .counts()
    }

    /// Writes the file when something changed since it was written: a new
    /// file beside it first, which then takes its place, so that the file
    /// is whole at every moment.
    pub fn save(&self) -> std::io::Result<()> {
        let Some(file) = &self.file else {
            return Ok(());
        };
        let text = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if !state.changed {
                return Ok(());
            }
            state.changed = false;
            state.written()
        };
        let new = file.with_extension("new");
        let done = std::fs::write(&new, text).and_then(|()| std::fs::rename(&new, file));
        if done.is_err() {
            // What was not written is written the next time.
            self.state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .changed = true;
        }
        done
    }

    pub fn file(&self) -> Option<&Path> {
        self.file.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("tet-web-counter-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory.join("counter")
    }

    #[test]
    fn requests_and_clients_are_counted_for_each_program() {
        let (counter, remark) = Counter::open(None);
        assert_eq!(remark, None);
        assert_eq!(counter.counts().clients, 0);
        counter.count(Program::Tet, "192.0.2.1");
        counter.count(Program::Tet, "192.0.2.1");
        counter.count(Program::Sm, "192.0.2.1");
        counter.count(Program::Sm, "192.0.2.2");
        counter.count(Program::Sm, " 192.0.2.2 ");
        let counts = counter.counts();
        assert_eq!((counts.tet, counts.sm), (2, 3));
        assert_eq!(
            (counts.clients, counts.tet_clients, counts.sm_clients),
            (2, 1, 2)
        );
        assert!(!counts.full && counts.since > 0);
        // Without a file there is nothing to write.
        assert!(counter.save().is_ok() && counter.file().is_none());
    }

    #[test]
    fn an_address_of_ipv6_counts_with_its_network_and_a_mapped_one_as_ipv4() {
        let (counter, _) = Counter::open(None);
        for client in [
            "2001:db8:1:2::1",
            "2001:db8:1:2:ffff:ffff:ffff:ffff",
            "2001:db8:1:3::1",
            "::ffff:192.0.2.7",
            "192.0.2.7",
            "not an address",
            "not an address",
        ] {
            counter.count(Program::Tet, client);
        }
        let counts = counter.counts();
        assert_eq!((counts.tet, counts.clients), (7, 4));
    }

    #[test]
    fn the_file_keeps_the_counts_and_no_address() {
        let file = scratch("kept");
        let (counter, remark) = Counter::open(Some(file.clone()));
        assert_eq!(remark, None);
        counter.count(Program::Tet, "198.51.100.23");
        counter.count(Program::Sm, "198.51.100.23");
        counter.count(Program::Sm, "2001:db8::5");
        counter.save().unwrap();
        let text = std::fs::read_to_string(&file).unwrap();
        assert!(text.starts_with("tet-web counter 1\nsince "), "{text}");
        assert!(text.contains("\ntet 1\nsm 2\n"), "{text}");
        assert_eq!(text.matches("\nclient ").count(), 2, "{text}");
        assert!(!text.contains("198.51") && !text.contains("2001"), "{text}");
        // Nothing changed: the file is not written again.
        std::fs::remove_file(&file).unwrap();
        counter.save().unwrap();
        assert!(!file.exists());
        counter.count(Program::Tet, "198.51.100.23");
        counter.save().unwrap();
        // Another server finds the counts, and knows the clients again.
        let (again, remark) = Counter::open(Some(file.clone()));
        assert_eq!(remark, None);
        assert_eq!(again.counts(), counter.counts());
        again.count(Program::Tet, "198.51.100.23");
        again.count(Program::Tet, "2001:db8::6");
        let counts = again.counts();
        assert_eq!((counts.tet, counts.sm), (4, 2));
        assert_eq!(
            (counts.clients, counts.tet_clients, counts.sm_clients),
            (2, 2, 2)
        );
        assert_eq!(counts.since, counter.counts().since);
        let _ = std::fs::remove_dir_all(file.parent().unwrap());
    }

    #[test]
    fn a_file_that_is_not_understood_is_set_aside() {
        let file = scratch("broken");
        for broken in [
            "something else\n",
            "tet-web counter 1\nsince 5\n",
            "tet-web counter 1\nsince 5\nsecret 1 2\ntet many\n",
            "tet-web counter 1\nsince 5\nsecret 1 2\nclient 0f 9\n",
            "tet-web counter 1\nsince 5\nsecret 1 2\nwhat is this\n",
        ] {
            std::fs::write(&file, broken).unwrap();
            let (counter, remark) = Counter::open(Some(file.clone()));
            let remark = remark.unwrap();
            assert!(remark.contains("is not understood"), "{remark}");
            assert!(remark.contains("counter.unread"), "{remark}");
            assert_eq!(counter.counts().tet, 0);
            assert_eq!(
                std::fs::read_to_string(file.with_extension("unread")).unwrap(),
                broken
            );
            assert!(!file.exists());
        }
        // What the server wrote itself is read back as it was.
        let state = State::read(
            "tet-web counter 1\nsince 7\nsecret a b\ntet 3\nsm 4\nfull\nclient 00000000000000ff 3\n",
        )
        .unwrap();
        assert_eq!(
            State::read(&state.written()).unwrap().written(),
            state.written()
        );
        assert!(state.counts().full);
        let _ = std::fs::remove_dir_all(file.parent().unwrap());
    }

    #[test]
    fn the_hash_of_a_client_is_the_same_everywhere_and_depends_on_the_secret() {
        assert_eq!(
            hashed((1, 2), &[192, 0, 2, 1]),
            hashed((1, 2), &[192, 0, 2, 1])
        );
        assert_ne!(
            hashed((1, 2), &[192, 0, 2, 1]),
            hashed((1, 3), &[192, 0, 2, 1])
        );
        assert_ne!(
            hashed((1, 2), &[192, 0, 2, 1]),
            hashed((1, 2), &[192, 0, 2, 2])
        );
        // A file written by one build must be read by the next: this value is fixed.
        assert_eq!(hashed((1, 2), &[192, 0, 2, 1]), 0xbc76_e77c_e363_950d);
        let (first, second) = (secret(), secret());
        assert_ne!(first, second);
        assert_ne!(first.0, first.1);
    }
}
