use std::{hash::Hash, marker::PhantomData};

use ahash::AHashSet;
use glam::{ivec2, ivec3, IVec2, IVec3};
use serde::{de::Visitor, ser::SerializeMap, Deserialize, Deserializer, Serialize, Serializer};

pub trait SerdeKey: Hash + Eq {
    fn to_key_string(&self) -> String;

    fn from_key_string(str: String) -> Self;
}

impl SerdeKey for IVec2 {
    fn to_key_string(&self) -> String {
        format!("{},{}", self.x, self.y)
    }

    fn from_key_string(str: String) -> Self {
        let (x, y) = str.split_once(',').unwrap();
        let (x, y) = (x.parse().unwrap(), y.parse().unwrap());
        ivec2(x, y)
    }
}

impl SerdeKey for IVec3 {
    fn to_key_string(&self) -> String {
        format!("{},{},{}", self.x, self.y, self.z)
    }

    fn from_key_string(str: String) -> Self {
        let mut parts = str.splitn(3, ',');
        let (x, y, z) = (
            parts.next().unwrap(),
            parts.next().unwrap(),
            parts.next().unwrap(),
        );
        let (x, y, z) = (x.parse().unwrap(), y.parse().unwrap(), z.parse().unwrap());
        ivec3(x, y, z)
    }
}

pub fn serialize<S: Serializer, K: SerdeKey>(t: &AHashSet<K>, s: S) -> Result<S::Ok, S::Error> {
    let mut map = s.serialize_map(Some(t.len()))?;
    for k in t {
        map.serialize_entry(&k.to_key_string(), &())?;
    }
    map.end()
}

pub fn deserialize<'de, D: Deserializer<'de>, K: SerdeKey>(d: D) -> Result<AHashSet<K>, D::Error> {
    struct V<K> {
        marker: PhantomData<fn(K)>,
    }
    impl<'de, K: SerdeKey> Visitor<'de> for V<K> {
        type Value = AHashSet<K>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a key-value map with stringified vectors as key and () as value")
        }

        fn visit_map<A>(self, mut set: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut res = AHashSet::with_capacity(set.size_hint().unwrap_or(0));
            while let Some((key, _)) = set.next_entry::<String, ()>()? {
                res.insert(K::from_key_string(key));
            }

            Ok(res)
        }
    }
    d.deserialize_map(V {
        marker: PhantomData::default(),
    })
}
